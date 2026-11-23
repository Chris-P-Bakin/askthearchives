import json
import pytest
from botocore.exceptions import ClientError

import common
import r2stage

SECRET = "sekrit-key-material-123"
KEYID = "AKIDEXAMPLE456"


class FakeClient:
    def __init__(self, existing=None, corrupt=False, fail_upload=False):
        self.objects = dict(existing or {})
        self.corrupt = corrupt
        self.fail_upload = fail_upload
        self.uploads = []

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "nf"}}, "HeadObject")
        return {"ContentLength": self.objects[Key]}

    def upload_file(self, Filename, Bucket, Key):
        if self.fail_upload:
            raise ClientError({"Error": {"Code": "500", "Message": "boom"}}, "PutObject")
        size = len(open(Filename, "rb").read())
        self.uploads.append(Key)
        self.objects[Key] = size + 1 if self.corrupt else size


@pytest.fixture
def units(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA", tmp_path)
    out = []
    for n in (1, 2, 3):
        d = tmp_path / "X1" / "pages"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{n}.png").write_bytes(b"png" * n * 10)
        out.append({"unit_id": f"X1-p{n}", "ntrs_id": "X1", "page": n,
                    "image_key": f"naca/pages/X1/p-{n}.png"})
    return out


def test_upload_then_verify(units):
    c = FakeClient()
    r = r2stage.stage_units(units, "b", c)
    assert (r["uploaded"], r["skipped"], r["failed"]) == (3, 0, 0)
    assert c.objects["naca/pages/X1/p-2.png"] == 60


def test_skip_if_exists(units):
    c = FakeClient(existing={"naca/pages/X1/p-1.png": 30})
    r = r2stage.stage_units(units, "b", c)
    assert (r["uploaded"], r["skipped"], r["failed"]) == (2, 1, 0)
    assert "naca/pages/X1/p-1.png" not in c.uploads


def test_existing_wrong_size_reuploaded(units):
    c = FakeClient(existing={"naca/pages/X1/p-1.png": 5})
    r = r2stage.stage_units(units[:1], "b", c)
    assert r["uploaded"] == 1 and c.objects["naca/pages/X1/p-1.png"] == 30


def test_size_mismatch_retried_once_then_failed(units):
    c = FakeClient(corrupt=True)
    r = r2stage.stage_units(units[:1], "b", c)
    assert (r["uploaded"], r["failed"]) == (0, 1)
    assert len(c.uploads) == 2


def test_upload_error_counts_failed(units):
    r = r2stage.stage_units(units, "b", FakeClient(fail_upload=True))
    assert r["failed"] == 3 and r["uploaded"] == 0


def test_missing_local_file_failed(units, tmp_path):
    (tmp_path / "X1" / "pages" / "1.png").unlink()
    r = r2stage.stage_units(units, "b", FakeClient())
    assert r["failed"] == 1 and r["uploaded"] == 2


def test_no_creds_error_names_vars_not_values(monkeypatch):
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(RuntimeError) as e:
        r2stage.client_from_env()
    assert "R2_ACCESS_KEY_ID" in str(e.value) and "R2_SECRET_ACCESS_KEY" in str(e.value)


def test_partial_creds_does_not_leak_value(monkeypatch):
    monkeypatch.setenv("R2_ACCESS_KEY_ID", KEYID)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(RuntimeError) as e:
        r2stage.client_from_env()
    assert KEYID not in str(e.value) and "R2_SECRET_ACCESS_KEY" in str(e.value)


def test_argv_and_logs_never_contain_secrets(units, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("R2_ACCESS_KEY_ID", KEYID)
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", SECRET)
    fake = FakeClient()
    monkeypatch.setattr(r2stage, "client_from_env", lambda: fake)
    uf = tmp_path / "u.jsonl"
    uf.write_text("\n".join(json.dumps(u) for u in units))
    argv = ["r2stage.py", "--units", str(uf), "--bucket", "b"]
    assert all(SECRET not in a and KEYID not in a for a in argv)
    assert r2stage.main(argv[1:]) == 0
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err and KEYID not in out.out + out.err
    assert "uploaded" in out.out


def test_dry_run_uploads_nothing(units, tmp_path, monkeypatch, capsys):
    fake = FakeClient(existing={"naca/pages/X1/p-1.png": 30})
    monkeypatch.setattr(r2stage, "client_from_env", lambda: fake)
    uf = tmp_path / "u.jsonl"
    uf.write_text("\n".join(json.dumps(u) for u in units))
    assert r2stage.main(["--units", str(uf), "--bucket", "b", "--dry-run"]) == 0
    assert fake.uploads == []
    assert "would upload" in capsys.readouterr().out


def test_main_nonzero_on_failure(units, tmp_path, monkeypatch):
    monkeypatch.setattr(r2stage, "client_from_env", lambda: FakeClient(fail_upload=True))
    uf = tmp_path / "u.jsonl"
    uf.write_text(json.dumps(units[0]))
    assert r2stage.main(["--units", str(uf), "--bucket", "b"]) == 1
