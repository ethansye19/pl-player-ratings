"""CLI output stays readable when it is redirected on Windows, where Python writes a redirected stdout in
cp1252: a name such as "Aleksandar Mitrović" ("ć" is not in cp1252) must not crash a command."""
import io
import os
import re
import subprocess
import sys

import pytest

from src.console import use_utf8_output
from tests.conftest import ROOT

NAME = "Aleksandar Mitrović"


def _cp1252_stream(errors: str = "strict"):
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding="cp1252", errors=errors, write_through=True)


def _cp1252_env() -> dict:
    """The encoding Windows gives a redirected stdout, made explicit so the test means the same everywhere."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONIOENCODING"] = "cp1252"
    return env


def test_a_cp1252_stream_cannot_take_the_name():
    _, stream = _cp1252_stream()
    with pytest.raises(UnicodeEncodeError):
        print(NAME, file=stream)


def test_use_utf8_output_lets_cp1252_stdout_and_stderr_take_the_name(monkeypatch):
    out_raw, out = _cp1252_stream()
    err_raw, err = _cp1252_stream(errors="backslashreplace")  # Python's own stderr handler
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    use_utf8_output()
    print(NAME)
    print(NAME, file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    assert out_raw.getvalue().decode("utf-8").splitlines() == [NAME]
    assert err_raw.getvalue().decode("utf-8").splitlines() == [NAME], "readable, not backslash-escaped"
    assert sys.stdout.errors == "strict" and sys.stderr.errors == "backslashreplace", "error handlers are kept"
    use_utf8_output()  # idempotent on streams that are already UTF-8
    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"


def test_use_utf8_output_leaves_streams_without_reconfigure_alone(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)  # pythonw
    monkeypatch.setattr(sys, "stderr", object())
    use_utf8_output()


@pytest.mark.parametrize("setup, works", [("", False), ("use_utf8_output(); ", True)])
def test_a_redirected_cp1252_process_prints_the_name_only_after_use_utf8_output(setup, works):
    code = f"from src.console import use_utf8_output; {setup}print({NAME!r})"
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=_cp1252_env(), capture_output=True, timeout=60)
    if works:
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.decode("utf-8").strip() == NAME
    else:
        assert proc.returncode != 0 and b"UnicodeEncodeError" in proc.stderr


def _cli_modules():
    return [p for p in sorted((ROOT / "src").rglob("*.py")) if 'if __name__ == "__main__":' in p.read_text(encoding="utf-8")]


def test_every_cli_switches_to_utf8_output_before_anything_else():
    found = {p.relative_to(ROOT).as_posix() for p in _cli_modules()}
    assert {"src/ingest/build.py", "src/explain/explain.py", "src/baseline/rate.py", "src/pipeline/gameweek.py"} <= found
    for path in _cli_modules():
        text = path.read_text(encoding="utf-8")
        main = re.search(r"^def main\(.*\).*:\n(.+)\n", text, flags=re.M)
        if main:
            assert main.group(1).strip() == "use_utf8_output()", f"{path}: main() must call use_utf8_output() first"
        else:
            block = text.split('if __name__ == "__main__":', 1)[1].splitlines()[1:5]
            assert any(line.strip() == "use_utf8_output()" for line in block), f"{path}: the __main__ block must call use_utf8_output()"


@pytest.mark.data
def test_explain_prints_mitrovic_when_its_output_is_redirected_in_cp1252():
    """Aleksandar Mitrović (player 4269) in Manchester City 6-1 Newcastle United, both methods."""
    proc = subprocess.run([sys.executable, "-m", "src.explain.explain", "--match", "3754079", "--player", "4269", "--method", "both"],
                          cwd=ROOT, env=_cp1252_env(), capture_output=True, timeout=300)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-2000:]
    text = proc.stdout.decode("utf-8")
    assert text.count(f"{NAME} (Newcastle United, FWD) rated") == 2
    assert "Mitrović's" in text
