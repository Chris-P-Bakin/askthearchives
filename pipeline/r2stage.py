"""Stage unit page images to Cloudflare R2 (S3 API). Credentials from env only."""
import argparse, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import units as units_mod

DEFAULT_ACCOUNT = os.environ.get("R2_ACCOUNT_ID", "")
CRED_VARS = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")


def client_from_env():
    missing = [v for v in CRED_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError("missing R2 credentials; set env var(s): " + ", ".join(missing))
    import boto3
    account = os.environ.get("R2_ACCOUNT_ID", DEFAULT_ACCOUNT)
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _remote_size(client, bucket, key):
    try:
        return client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except Exception as e:
        code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def _local_path(u):
    return units_mod.unit_doc_dir(u) / "pages" / f"{u['page']}.png"


def _stage_one(u, bucket, client, dry_run=False):
    """Return 'uploaded' | 'skipped' | 'failed' | 'would_upload'."""
    key = u["image_key"]
    try:
        path = _local_path(u)
        size = path.stat().st_size
        if _remote_size(client, bucket, key) == size:
            return "skipped"
        if dry_run:
            return "would_upload"
        for _ in range(2):  # one retry on failed upload or size mismatch
            try:
                client.upload_file(str(path), bucket, key)
                if _remote_size(client, bucket, key) == size:
                    return "uploaded"
            except Exception:
                pass
        return "failed"
    except Exception:
        return "failed"


def stage_units(units, bucket, client, workers=4, dry_run=False):
    res = {"uploaded": 0, "skipped": 0, "failed": 0, "would_upload": 0,
           "bytes": 0, "failed_keys": [], "would_upload_keys": []}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        outcomes = list(ex.map(lambda u: _stage_one(u, bucket, client, dry_run), units))
    for u, o in zip(units, outcomes):
        res[o] += 1
        if o == "uploaded":
            res["bytes"] += _local_path(u).stat().st_size
        elif o == "failed":
            res["failed_keys"].append(u["image_key"])
        elif o == "would_upload":
            res["would_upload_keys"].append(u["image_key"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    us = units_mod.read_units(a.units)
    try:
        client = client_from_env()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    r = stage_units(us, a.bucket, client, a.workers, a.dry_run)
    for k in r["would_upload_keys"]:
        print(f"would upload {k}")
    for k in r["failed_keys"]:
        print(f"FAILED {k}")
    print(f"units={len(us)} uploaded={r['uploaded']} skipped={r['skipped']} "
          f"failed={r['failed']} would_upload={r['would_upload']} bytes={r['bytes']}")
    return 1 if r["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
