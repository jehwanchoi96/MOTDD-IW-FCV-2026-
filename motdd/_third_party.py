"""Access to the two upstream code bases used by MOTDD.

* ByteTrack (MIT): Kalman filter, ``STrack``, ``BYTETracker`` and the matching
  helpers are imported from ``<ByteTrack>/yolox/tracker``.
* YOLOX (Apache-2.0): the detector re-invoked by PGR is built from
  ``<YOLOX>/yolox``.

Both repositories ship a top-level package called ``yolox``. They are
therefore never imported at the same time: the YOLOX package is imported
first (detector construction) and ByteTrack's package is imported afterwards
for every new tracker instance, after removing any cached ``yolox`` modules.
Re-importing ByteTrack per tracker instance also resets the global track-ID
counter, so track IDs start from 1 in every sequence.

Locations are taken from the environment variables ``MOTDD_BYTETRACK_DIR`` and
``MOTDD_YOLOX_DIR`` and default to ``third_party/ByteTrack`` and
``third_party/YOLOX`` inside this repository (see ``third_party/README.md``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def bytetrack_dir() -> Path:
    return Path(os.environ.get("MOTDD_BYTETRACK_DIR",
                               REPO_ROOT / "third_party" / "ByteTrack")).resolve()


def yolox_dir() -> Path:
    return Path(os.environ.get("MOTDD_YOLOX_DIR",
                               REPO_ROOT / "third_party" / "YOLOX")).resolve()


def _check_dir(path: Path, what: str) -> None:
    if not (path / "yolox").is_dir():
        raise FileNotFoundError(
            f"{what} not found at {path} (expected a 'yolox' package inside). "
            f"See third_party/README.md."
        )


def install_numpy_aliases() -> None:
    """Restore the ``np.float`` / ``np.int`` / ... aliases removed in NumPy 1.24.

    ByteTrack (and TrackEval) reference them; the aliases are installed on the
    ``numpy`` module object so the upstream sources stay unmodified.
    """
    import warnings

    import numpy as np

    aliases = {"float": float, "int": int, "bool": bool, "object": object,
               "float_": np.float64}
    with warnings.catch_warnings():
        # NumPy 1.2x emits a FutureWarning when probing the removed names.
        warnings.simplefilter("ignore", FutureWarning)
        for name, value in aliases.items():
            if not hasattr(np, name):
                setattr(np, name, value)


def ensure_yolox_on_path() -> None:
    """Put the YOLOX repository first on ``sys.path`` (detector side)."""
    root = yolox_dir()
    _check_dir(root, "YOLOX")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_bytetrack() -> Dict[str, Any]:
    """(Re-)import the ByteTrack tracker symbols.

    Any cached ``yolox`` modules are dropped first so that ByteTrack's own
    ``yolox`` package is imported, and so that ``BaseTrack``'s track-ID
    counter starts from zero for the new tracker instance.
    """
    install_numpy_aliases()
    root = bytetrack_dir()
    _check_dir(root, "ByteTrack")
    for name in list(sys.modules.keys()):
        if name.split(".", 1)[0] == "yolox":
            del sys.modules[name]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from yolox.tracker.byte_tracker import (  # type: ignore
        BYTETracker,
        STrack,
        joint_stracks,
        remove_duplicate_stracks,
        sub_stracks,
    )
    from yolox.tracker.basetrack import TrackState  # type: ignore
    from yolox.tracker import matching  # type: ignore

    return {
        "BYTETracker": BYTETracker,
        "STrack": STrack,
        "TrackState": TrackState,
        "matching": matching,
        "joint_stracks": joint_stracks,
        "sub_stracks": sub_stracks,
        "remove_duplicate_stracks": remove_duplicate_stracks,
    }
