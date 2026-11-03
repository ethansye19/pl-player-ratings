"""UTF-8 output for the command-line entry points (``python -m src.<area>.<module>``).

On Windows, Python writes a redirected or piped stdout (``> file``, ``| tee``, a log captured by a
scheduler) in the ANSI code page, usually cp1252. cp1252 has no "ć", "č", "ş" or "ł", so printing a
name such as "Aleksandar Mitrović" raised ``UnicodeEncodeError`` and the command exited 1 after its
work was done (``src.ingest.build`` did exactly that on its closing summary). Every CLI calls
:func:`use_utf8_output` first: stdout and stderr then write UTF-8 whatever the code page, so names stay
readable in the file. A console is already UTF-8, so nothing changes there.
"""
from __future__ import annotations

import sys


def _is_utf8(encoding: str | None) -> bool:
    return (encoding or "").lower().replace("-", "").replace("_", "") == "utf8"


def use_utf8_output() -> None:
    """Switch ``sys.stdout`` and ``sys.stderr`` to UTF-8, keeping their error handlers, unless they already are.

    Streams without ``reconfigure`` (None under pythonw, or a replacement object) are left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None or _is_utf8(getattr(stream, "encoding", None)):
            continue
        reconfigure(encoding="utf-8", errors=getattr(stream, "errors", None) or "strict")
