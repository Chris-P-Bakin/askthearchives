"""Submit NACA units to the BOINC project as workunits.

    python pipeline/boinc_submit.py data/units/batch_000_units.jsonl --limit 10

For each unit: copy unit.json + the page PNG into the staging dir (bind-mounted
at /naca/stage in the apache container), `bin/stage_file --copy` them into the
project's download hierarchy, then `bin/create_work` (min_quorum=1,
target_nresults=1: trusted-workers phase). Already-submitted units (WU
`naca_<unit_id>` in the DB) are skipped, so re-runs are idempotent.
All server commands run via `docker exec` into the stack's apache container.
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import units as units_mod
APP = "naca_vision"
CONTAINER = "naca-boinc-apache"
MYSQL_CONTAINER = CONTAINER          # mysql client lives in the apache image
PROJECT_DIR = "/home/boincadm/project"
STAGE_CONTAINER_DIR = "/naca/stage"
DEFAULT_STAGE = ROOT / "boinc" / "stage"


class StageError(Exception):
    pass


def wu_name(unit_id: str) -> str:
    return f"naca_{unit_id}"


def phys_names(unit_id: str) -> tuple[str, str]:
    return f"{unit_id}.unit.json", f"{unit_id}.page.png"


def create_work_args(unit: dict, name: str | None = None) -> list[str]:
    uid = unit["unit_id"]
    j, p = phys_names(uid)
    return ["bin/create_work", "--appname", APP, "--wu_name", name or wu_name(uid),
            "--wu_template", f"templates/{APP}_in",
            "--result_template", f"templates/{APP}_out",
            "--min_quorum", "1", "--target_nresults", "1", j, p]


def stage_unit(unit: dict, data_root, stage_dir) -> tuple[Path, Path]:
    """Write <uid>.unit.json and copy the page PNG into stage_dir (sha-checked)."""
    data_root, stage_dir = Path(data_root), Path(stage_dir)
    src = units_mod.unit_doc_dir(unit, data_root) / "pages" / f"{unit['page']}.png"
    try:
        img = src.read_bytes()
    except OSError:
        raise StageError(f"missing page image {src}")
    if hashlib.sha256(img).hexdigest() != unit["image_sha256"]:
        raise StageError(f"sha256 mismatch for {src}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    jn, pn = phys_names(unit["unit_id"])
    jp, pp = stage_dir / jn, stage_dir / pn
    jp.write_text(json.dumps(unit, sort_keys=True))
    pp.write_bytes(img)
    return jp, pp


# --- batched app (naca_vision_b): one WU = N pages, Windows native worker path ---
APP_B = "naca_vision_b"
MODEL_PHYS = "qwen3vl8b_lora_v2_1_q8.gguf"      # sticky, staged once (install step)
MMPROJ_PHYS = "qwen3vl8b_mmproj_f16.gguf"


def batch_wu_name(units: list[dict], gt: bool = False) -> str:
    """Deterministic batch name: naca_b_[gt_]<sha1-12 of the ordered unit ids>."""
    h = hashlib.sha1("+".join(u["unit_id"] for u in units).encode()).hexdigest()[:12]
    return f"naca_b_{'gt_' if gt else ''}{h}"


def build_batch_template(n: int) -> str:
    """Input template for an N-page batch: manifest + 2 sticky model refs + N pages."""
    infos, refs = [], []
    def ref(num, open_name, ref_extra="", info_extra=""):
        # sticky/no_delete are honored only in <file_info>; copy_file only in <file_ref>
        infos.append("    <file_info>\n"
                     f"        <number>{num}</number>\n"
                     + (f"        {info_extra}\n" if info_extra else "")
                     + "    </file_info>")
        refs.append("        <file_ref>\n"
                    f"            <file_number>{num}</file_number>\n"
                    f"            <open_name>{open_name}</open_name>\n"
                    + (f"            {ref_extra}\n" if ref_extra else "")
                    + "        </file_ref>")
    ref(0, "manifest.json", ref_extra="<copy_file/>")
    ref(1, "model.gguf", info_extra="<sticky/>\n        <no_delete/>")
    ref(2, "mmproj.gguf", info_extra="<sticky/>\n        <no_delete/>")
    for i in range(n):
        ref(3 + i, f"page_{i:03d}.png", ref_extra="<copy_file/>")
    return ("<input_template>\n" + "\n".join(infos) + "\n    <workunit>\n"
            + "\n".join(refs) + "\n"
            "        <rsc_fpops_est>3e14</rsc_fpops_est>\n"
            "        <rsc_fpops_bound>3e17</rsc_fpops_bound>\n"
            "        <rsc_memory_bound>2e9</rsc_memory_bound>\n"
            "        <rsc_disk_bound>2e9</rsc_disk_bound>\n"
            "        <delay_bound>172800</delay_bound>\n"
            "        <min_quorum>1</min_quorum>\n"
            "        <target_nresults>1</target_nresults>\n"
            "        <max_error_results>4</max_error_results>\n"
            "        <max_total_results>6</max_total_results>\n"
            "        <max_success_results>1</max_success_results>\n"
            "    </workunit>\n</input_template>\n")


def batch_create_work_args(units: list[dict], gt: bool = False) -> list[str]:
    name = batch_wu_name(units, gt)
    pages = [phys_names(u["unit_id"])[1] for u in units]
    return ["bin/create_work", "--appname", APP_B, "--wu_name", name,
            "--wu_template", f"templates/{APP_B}_in_{len(units)}",
            "--result_template", f"templates/{APP_B}_out",
            "--min_quorum", "1", "--target_nresults", "1",
            f"{name}.manifest.json", MODEL_PHYS, MMPROJ_PHYS, *pages]


def _exec(container: str, cmd: list[str], user="boincadm", workdir=PROJECT_DIR) -> list[str]:
    return ["docker", "exec", "-u", user, "-w", workdir, container, *cmd]


def existing_wu_names(run, container: str) -> set[str]:
    sql = "SELECT name FROM workunit WHERE name LIKE 'naca\\_%' AND error_mask=0"
    script = ('source /run/secrets/secrets.env; '
              f'mysql -h mysql -uroot -p"$DB_PASSWD" naca -N -e "{sql}"')
    r = run(["docker", "exec", container, "bash", "-c", script],
            capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cannot list existing workunits: {r.stderr}")
    return {l.strip() for l in r.stdout.splitlines() if l.strip()}


def stage_into_project(unit, data_root, stage_dir, run=subprocess.run, container=CONTAINER):
    """Stage unit.json + page PNG (sha-checked) and register them in the download hierarchy."""
    staged = stage_unit(unit, data_root, stage_dir)
    try:
        for f in staged:
            r = run(_exec(container, ["bin/stage_file", "--copy",
                                      f"{STAGE_CONTAINER_DIR}/{f.name}"]),
                    capture_output=True, text=True)
            if r.returncode != 0:
                raise StageError(f"stage_file {f.name}: {r.stderr.strip()}")
    finally:
        for f in staged:
            f.unlink(missing_ok=True)


def submit(units, data_root, stage_dir, run=subprocess.run, container=CONTAINER,
           limit=0, dry_run=False) -> dict:
    stage_dir = Path(stage_dir)
    counts = {"submitted": 0, "skipped_existing": 0, "failed": 0}
    if dry_run:
        return counts
    existing = existing_wu_names(run, container)
    todo = []
    for u in units:
        if wu_name(u["unit_id"]) in existing:
            counts["skipped_existing"] += 1
        else:
            todo.append(u)
    if limit:
        todo = todo[:limit]
    for u in todo:
        try:
            stage_into_project(u, data_root, stage_dir, run, container)
            r = run(_exec(container, create_work_args(u)), capture_output=True, text=True)
            if r.returncode != 0:
                raise StageError(f"create_work {u['unit_id']}: {r.stderr.strip()}")
        except StageError as e:
            counts["failed"] += 1
            print(f"{u['unit_id']}: FAILED {e}", file=sys.stderr)
            continue
        counts["submitted"] += 1
        print(f"{u['unit_id']}: submitted", flush=True)
    return counts


def submit_batched(units, data_root, stage_dir, run=subprocess.run, container=CONTAINER,
                   batch_n=32, gt=False, limit=0, dry_run=False) -> dict:
    """Group units into N-page naca_vision_b workunits and create them (idempotent)."""
    stage_dir = Path(stage_dir)
    counts = {"submitted": 0, "skipped_existing": 0, "failed": 0}
    if limit:
        units = units[:limit]
    batches = [units[i:i + batch_n] for i in range(0, len(units), batch_n)]
    if dry_run:
        return counts
    existing = existing_wu_names(run, container)
    for batch in batches:
        name = batch_wu_name(batch, gt)
        if name in existing:
            counts["skipped_existing"] += 1
            continue
        try:
            staged = [stage_unit(u, data_root, stage_dir)[1] for u in batch]  # pages only
            manifest = stage_dir / f"{name}.manifest.json"
            manifest.write_text(json.dumps(batch, sort_keys=True))
            tmpl = stage_dir / f"{APP_B}_in_{len(batch)}"
            tmpl.write_text(build_batch_template(len(batch)))
            try:
                for f in (*staged, manifest):
                    r = run(_exec(container, ["bin/stage_file", "--copy",
                                              f"{STAGE_CONTAINER_DIR}/{f.name}"]),
                            capture_output=True, text=True)
                    if r.returncode != 0:
                        raise StageError(f"stage_file {f.name}: {r.stderr.strip()}")
                r = run(_exec(container, ["cp", f"{STAGE_CONTAINER_DIR}/{tmpl.name}",
                                          f"templates/{tmpl.name}"]),
                        capture_output=True, text=True)
                if r.returncode != 0:
                    raise StageError(f"template install: {r.stderr.strip()}")
                r = run(_exec(container, batch_create_work_args(batch, gt)),
                        capture_output=True, text=True)
                if r.returncode != 0:
                    raise StageError(f"create_work {name}: {r.stderr.strip()}")
            finally:
                for f in (*staged, manifest, tmpl,
                          *(stage_dir / phys_names(u["unit_id"])[0] for u in batch)):
                    f.unlink(missing_ok=True)
        except StageError as e:
            counts["failed"] += 1
            print(f"{name}: FAILED {e}", file=sys.stderr)
            continue
        counts["submitted"] += 1
        print(f"{name}: submitted ({len(batch)} pages)", flush=True)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("units")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--stage-dir", default=str(DEFAULT_STAGE))
    ap.add_argument("--container", default=CONTAINER)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--app", choices=[APP, APP_B], default=APP)
    ap.add_argument("--batch", type=int, default=32, help="pages per WU (--app naca_vision_b)")
    ap.add_argument("--gt", action="store_true", help="GT acceptance batch: route to vision_8bwin")
    a = ap.parse_args(argv)
    units = [json.loads(l) for l in open(a.units) if l.strip()]
    if a.app == APP_B:
        c = submit_batched(units, a.data_root, a.stage_dir, container=a.container,
                           batch_n=a.batch, gt=a.gt, limit=a.limit, dry_run=a.dry_run)
    else:
        c = submit(units, a.data_root, a.stage_dir, container=a.container,
                   limit=a.limit, dry_run=a.dry_run)
    print(json.dumps(c))
    return 1 if c["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
