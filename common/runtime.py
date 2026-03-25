"""Runtime helpers shared across NNUI2 experiments."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def configure_matplotlib_env(cache_name: str = "nnui2") -> None:
    """Point matplotlib/fontconfig caches to writable temp locations."""

    root = Path(tempfile.gettempdir()) / cache_name
    mpl_dir = root / "mpl"
    xdg_dir = root / "xdg"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))
