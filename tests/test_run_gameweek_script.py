"""scripts/run_gameweek.ps1, the scheduled-task wrapper: exit codes, the log, and the failure notification.

Each test copies the real script into a temporary tree with its own small venv and a SYNTHETIC stand-in
for src.pipeline.gameweek, so nothing is published and no network is used. RATING_SYSTEM_NOTIFY=log makes
the script write the notification to the log instead of showing it. Windows only (Windows PowerShell)."""
import os
import shutil
import subprocess
import sys

import pytest

from tests.conftest import ROOT

pytestmark = pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell.exe") is None,
                                reason="the wrapper is a Windows PowerShell script")

STUB = '''"""SYNTHETIC stand-in for src.pipeline.gameweek, used only to exercise scripts/run_gameweek.ps1."""
import os
import sys

mode = os.environ["STUB_MODE"]
if mode == "published":
    print("gameweek 10: 280 rated player-fixtures in 10 fixtures, mean 6.55; wrote 13 files to web/data/current")
elif mode == "nothing":
    print("nothing to do: latest finished gameweek 10 is already published in web/data/current")
elif mode == "checks":
    print("ERROR check failed: gameweek 10: Synthetic Player (player 7): the per-fixture 'minutes' add up to 95 but the gameweek total is 90", file=sys.stderr)
    print("gameweek 10: checks FAILED, nothing written")
    print("  - gameweek 10: Synthetic Player (player 7): the per-fixture 'minutes' add up to 95 but the gameweek total is 90")
    sys.exit(1)
elif mode == "crash":
    raise RuntimeError("FPL API request failed: https://fantasy.premierleague.com/api/bootstrap-static/: read timed out")
'''
CHECK_FAILURE = "gameweek 10: Synthetic Player (player 7): the per-fixture 'minutes' add up to 95 but the gameweek total is 90"


@pytest.fixture(scope="module")
def venv_template(tmp_path_factory):
    path = tmp_path_factory.mktemp("venv") / ".venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(path)], check=True, timeout=300)
    return path


def _tree(root, venv_template, with_python=True):
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts" / "run_gameweek.ps1", root / "scripts" / "run_gameweek.ps1")
    pkg = root / "src" / "pipeline"
    pkg.mkdir(parents=True)
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "gameweek.py").write_text(STUB, encoding="utf-8")
    if with_python:
        shutil.copytree(venv_template, root / ".venv")
    return root


def _run(tree, mode, notify="log", extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "RATING_SYSTEM_NOTIFY", "RATING_SYSTEM_PUSH_KEY")}
    env.update(STUB_MODE=mode, **(extra_env or {}))
    if notify is not None:
        env["RATING_SYSTEM_NOTIFY"] = notify
    proc = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(tree / "scripts" / "run_gameweek.ps1")],
                          env=env, capture_output=True, timeout=300)
    log = (tree / "logs" / "pipeline.log").read_text(encoding="utf-8-sig")
    return proc.returncode, log


@pytest.mark.parametrize("mode", ["published", "nothing"])
def test_exit_0_is_silent(tmp_path, venv_template, mode):
    code, log = _run(_tree(tmp_path, venv_template), mode)
    assert code == 0
    assert log.rstrip().endswith("run_gameweek end (exit 0)")
    assert "notification" not in log


def test_failed_checks_exit_1_and_notify_with_the_first_failure(tmp_path, venv_template):
    tree = _tree(tmp_path, venv_template)
    code, log = _run(tree, "checks")
    assert code == 1
    assert "checks FAILED, nothing written" in log and "run_gameweek end (exit 1)" in log
    assert f"notification (not shown, RATING_SYSTEM_NOTIFY=log): Rating pipeline failed (exit 1) | {CHECK_FAILURE} | {tree / 'logs' / 'pipeline.log'}" in log
    assert log.index("run_gameweek end (exit 1)") < log.index("notification"), "the log is complete before the notification"


def test_a_python_crash_exits_1_and_notifies_with_the_exception(tmp_path, venv_template):
    code, log = _run(_tree(tmp_path, venv_template), "crash")
    assert code == 1
    assert "Traceback (most recent call last)" in log
    assert ("Rating pipeline failed (exit 1) | RuntimeError: FPL API request failed: "
            "https://fantasy.premierleague.com/api/bootstrap-static/: read timed out") in log


def test_a_missing_venv_python_exits_1_and_notifies(tmp_path, venv_template):
    tree = _tree(tmp_path, venv_template, with_python=False)
    code, log = _run(tree, "published")
    assert code == 1
    assert f"Rating pipeline failed (exit 1) | venv python not found at {tree / '.venv' / 'Scripts' / 'python.exe'}" in log
    assert "run_gameweek end (exit 1)" in log


@pytest.mark.parametrize("mode", ["checks", "crash"])
def test_a_notification_that_fails_changes_neither_the_exit_code_nor_the_log(tmp_path, venv_template, mode):
    code, log = _run(_tree(tmp_path, venv_template), mode, notify="fail")
    assert code == 1
    assert "run_gameweek end (exit 1)" in log
    assert "notification could not be shown: simulated notification failure" in log


@pytest.fixture
def notifications_switched_off():
    """A temporary HKCU registry key standing in for Windows' PushNotifications key, with ToastEnabled = 0
    (the main notifications switch off). The real setting is never read or changed; the key is removed afterwards."""
    import uuid
    import winreg

    parent = r"Software\RatingSystemTests"
    run_key = rf"{parent}\{uuid.uuid4().hex}"
    path = rf"{run_key}\PushNotifications"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, path) as key:
        winreg.SetValueEx(key, "ToastEnabled", 0, winreg.REG_DWORD, 0)
    yield {"RATING_SYSTEM_PUSH_KEY": "HKCU:\\" + path}
    for sub in (path, run_key, parent):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, sub)
        except OSError:
            pass  # the parent stays if another test run is using it


@pytest.mark.parametrize("mode", ["checks", "crash"])
def test_switched_off_notifications_are_logged_as_not_displayed(tmp_path, venv_template, notifications_switched_off, mode):
    """Notifications on in the script (no test mode), but the Windows switch is off: the log must not claim 'shown'."""
    code, log = _run(_tree(tmp_path, venv_template), mode, notify=None, extra_env=notifications_switched_off)
    assert code == 1, "the exit code does not change"
    assert "notification not displayed: Windows notifications are switched off" in log
    assert "Rating pipeline failed (exit 1)" in log.split("not displayed")[1]
    assert "notification shown" not in log and "could not be shown" not in log
    assert log.index("run_gameweek end (exit 1)") < log.index("not displayed")


def test_a_missing_venv_python_with_notifications_switched_off_still_exits_1(tmp_path, venv_template, notifications_switched_off):
    tree = _tree(tmp_path, venv_template, with_python=False)
    code, log = _run(tree, "published", notify=None, extra_env=notifications_switched_off)
    assert code == 1
    assert "notification not displayed: Windows notifications are switched off" in log and "venv python not found" in log
