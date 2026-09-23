"""Put every scripts/<subdir> on sys.path so reorganized modules still import."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
SKIP = {"__pycache__", ".git"}

for p in sorted(SCRIPTS.iterdir(), key=lambda x: x.name, reverse=True):
    if p.is_dir() and p.name not in SKIP:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
