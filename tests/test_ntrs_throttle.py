import subprocess
import sys
import time

import common


def test_ntrs_wait_cold_start(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    lock = tmp_path / "ntrs.lock"
    common.ntrs_wait(4.0, lock_path=lock)
    assert sleeps == []
    assert lock.exists()


def test_ntrs_wait_spaces_requests(tmp_path, monkeypatch):
    now = [1000.0]
    sleeps = []
    monkeypatch.setattr(time, "time", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)))
    lock = tmp_path / "ntrs.lock"
    common.ntrs_wait(4.0, lock_path=lock)
    now[0] += 1.5
    common.ntrs_wait(4.0, lock_path=lock)
    assert len(sleeps) == 1 and abs(sleeps[0] - 2.5) < 1e-6


def test_ntrs_wait_cross_process(tmp_path):
    lock = tmp_path / "ntrs.lock"
    code = ("import sys; sys.path.insert(0, %r); import common; "
            "common.ntrs_wait(0.3, lock_path=%r)" % (str(common.ROOT / "pipeline"), str(lock)))
    subprocess.run([sys.executable, "-c", code], check=True)
    t0 = time.monotonic()
    common.ntrs_wait(0.3, lock_path=lock)
    assert time.monotonic() - t0 >= 0.2
