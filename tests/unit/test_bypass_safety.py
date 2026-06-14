"""Lock-1 measurement-bypass containment guard (Knight-Capital lesson, SEC 34-70694).

The ``allow_same_family=True`` bypass is the ONLY switch that disables Lock 1; it exists solely for
the ``prism eval --family-ab`` same-family control arm. A guard/flag bypass that silently leaks into
a production path is precisely the failure mode that reactivated dead code and lost $460M. This scan
fails CI if the bypass is enabled anywhere outside its two legitimate sites:

  * ``cli/main.py``    — the ``_build_same_family_control`` measurement arm (the one allowed USE),
  * ``core/routing.py`` — where the flag is DEFINED + documented (its module docstring names it).

Any other production module (engine / http / mcp / setup / the verify path) enabling it is a defect.
The default-off behavior is separately locked by ``test_routing_same_family_bypass.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "prism"
_BYPASS = re.compile(r"allow_same_family\s*=\s*True")
_ALLOWED = ["cli/main.py", "core/routing.py"]


def test_allow_same_family_bypass_confined_to_eval_control() -> None:
    offenders = sorted(
        py.relative_to(_SRC).as_posix()
        for py in _SRC.rglob("*.py")
        if _BYPASS.search(py.read_text(encoding="utf-8"))
    )
    assert offenders == _ALLOWED, (
        "Lock-1 measurement bypass allow_same_family=True must appear ONLY in the --family-ab "
        f"control (cli/main.py) and its definition module (core/routing.py). Found in: {offenders}"
    )
