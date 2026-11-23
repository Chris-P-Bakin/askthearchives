"""Select the 30 tuning pages: hard page types, per the spec.

Deterministic given data/: uses baseline stats (dict share, flags), digit
density from the baseline text (table proxy), and the CSV's pilot notes.
Writes data/tuning_pages.csv. The 70 reporting pages are drawn separately,
proportional to page count, and must never overlap this set.
"""
import csv, json, re
import common

def page_features(ntrs_id):
    d = common.report_dir(ntrs_id)
    stats = json.load(open(d / "baseline" / "stats.json"))
    feats = {}
    for p, s in stats["pages"].items():
        txt = (d / "baseline" / f"{p}.txt").read_text(errors="replace")
        alnum = sum(c.isalnum() for c in txt)
        feats[int(p)] = {**s, "eq": txt.count("="),
                         "digit_ratio": round(sum(c.isdigit() for c in txt) / alnum, 3) if alnum else 0.0}
    return feats

def main():
    rows = {r["ntrs_id"]: r for r in common.load_reports()}
    F = {nid: page_features(nid) for nid in rows}
    picked, used = [], {}          # used: pages per report

    def take(nid, page, category, focus):
        if used.get(nid, 0) >= 2 or any(p[0] == nid and p[1] == page for p in picked):
            return False
        f = F[nid][page]
        picked.append((nid, page, category, focus, f))
        used[nid] = used.get(nid, 0) + 1
        return True

    def best(nids, key, flt, category, focus, n=1):
        cands = [(key(f), nid, pg, f) for nid in nids for pg, f in F[nid].items() if flt(f)]
        got = 0
        for _, nid, pg, f in sorted(cands, key=lambda c: c[0]):
            if got >= n: break
            if take(nid, pg, category, focus): got += 1
        return got

    by_series = lambda s: [nid for nid, r in rows.items() if r["series"] == s]
    by_report = {r["report_number"]: nid for nid, r in rows.items()}
    text = lambda f: f["chars"] > 800 and "flag" not in f

    # 1 exemplar: TR-96 interior title page (known garble: LONGH'LUXNAL, WARXER)
    take(by_report["NACA-TR-96"], 3, "exemplar", "typeset 1921; known baseline garble")
    # 4 dense numeric tables: the CSV-noted reports + the densest RM table page
    for rep in ("NACA-TR-218", "NACA-TR-1037", "NACA-TR-1135"):
        best([by_report[rep]], lambda f: -f["digit_ratio"], text, "table", "dense numeric table")
    best(by_series("RM"), lambda f: -f["digit_ratio"], text, "table", "dense numeric table (RM)")
    # 3 equations: TR-452 + TR-1135 + lowest-share remaining TR equationish page
    eqn = lambda f: f["chars"] > 400 and f["eq"] >= 10 and "flag" not in f
    best([by_report["NACA-TR-452"]], lambda f: -f["eq"], eqn, "equation", "heavy equations")
    best([by_report["NACA-TR-1135"]], lambda f: -f["eq"], eqn, "equation", "equations stress test")
    best(by_series("TR"), lambda f: -f["eq"], eqn, "equation", "equation-dense TR page")
    # 6 typewritten wartime, lowest dict share, distinct reports
    for _ in range(6):
        best(by_series("WR"), lambda f: f["dict_share"], text, "typewritten", "typewritten wartime; low dict share")
    # 2 RM low-share (charts + supersonic vocabulary)
    for _ in range(2):
        best(by_series("RM"), lambda f: f["dict_share"], text, "rm-hard", "1950s charts; low dict share")
    # 2 worst TN pages (worst reports in the corpus by dict share)
    best([by_report["NACA-TN-2944"]], lambda f: f["dict_share"], text, "tn-hard", "worst-share report")
    best([by_report["NACA-TN-243"]], lambda f: f["dict_share"], lambda f: f["chars"] > 500 and "flag" not in f, "tn-hard", "worst-share 1920s TN")
    # 3 TM translations (foreign names), lowest share, distinct reports
    for _ in range(3):
        best(by_series("TM"), lambda f: f["dict_share"],
             lambda f: text(f) and f["dict_share"] > 0.25, "translation", "translation; foreign names")
    # 3 photo-heavy AC: two near-image-only caption pages + one low-share text page
    cap = lambda f: f.get("flag") == "image_only"
    best(by_series("AC"), lambda f: -f["chars"], cap, "caption", "photo page, caption only", n=2)
    best(by_series("AC"), lambda f: f["dict_share"], text, "ac-text", "illustrated circular text")
    # 4 CONF: per volume, densest table page + lowest-share text page
    for nid in by_series("CONF"):
        best([nid], lambda f: -f["digit_ratio"], text, "table", "conference volume table")
        best([nid], lambda f: f["dict_share"], text, "conf-text", "conference volume text")
    # 1 noise page: blank scan where the 1990s layer hallucinated text
    best(list(rows), lambda f: f["dict_share"],
         lambda f: f["chars"] > 800 and f["dict_share"] < 0.1 and "flag" not in f,
         "noise", "blank scan; baseline hallucinated text")
    # fill to 30: lowest dict share anywhere, favouring under-represented decades
    while len(picked) < 30:
        decades = {}
        for nid, pg, *_ in picked:
            decades[rows[nid]["decade"]] = decades.get(rows[nid]["decade"], 0) + 1
        def fill_key(item):
            nid, f = item[0], item[2]
            return (decades.get(rows[nid]["decade"], 0), f["dict_share"])
        cands = sorted(((nid, pg, f) for nid in rows for pg, f in F[nid].items() if text(f)), key=fill_key)
        for nid, pg, f in cands:
            if take(nid, pg, "fill", "low dict share; decade balance"):
                break
    # write
    out = common.DATA / "tuning_pages.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ntrs_id", "report_number", "series", "decade", "pdf_page",
                    "category", "focus", "chars", "dict_share", "digit_ratio",
                    "image", "baseline_text"])
        for nid, pg, cat, focus, f in sorted(picked, key=lambda p: (rows[p[0]]["series"], p[0], p[1])):
            r = rows[nid]
            w.writerow([nid, r["report_number"], r["series"], r["decade"], pg, cat, focus,
                        f["chars"], f["dict_share"], f["digit_ratio"],
                        f"data/{nid}/pages/{pg}.png", f"data/{nid}/baseline/{pg}.txt"])
    print(f"wrote {out} with {len(picked)} pages")
    cov_s, cov_d = {}, {}
    for nid, *_ in picked:
        cov_s[rows[nid]["series"]] = cov_s.get(rows[nid]["series"], 0) + 1
        cov_d[rows[nid]["decade"]] = cov_d.get(rows[nid]["decade"], 0) + 1
    print("series:", dict(sorted(cov_s.items())), "| decades:", dict(sorted(cov_d.items())))

if __name__ == "__main__":
    main()
