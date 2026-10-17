"""Plain ByteTrack baseline (first row of the ablation): the upstream ``BYTETracker``."""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np

from .._third_party import load_bytetrack

Row = Tuple[int, float, float, float, float, float]


class ByteTrackBaseline:
    """Thin adapter around the official ``BYTETracker`` with the MOTDD tracker API."""

    def __init__(self, params: dict, frame_rate: int = 30) -> None:
        syms = load_bytetrack()
        args = SimpleNamespace(
            track_thresh=float(params.get("track_thresh", 0.5)),
            track_buffer=int(params.get("track_buffer", 30)),
            match_thresh=float(params.get("match_thresh", 0.8)),
            mot20=bool(params.get("mot20", False)),
        )
        self._tracker = syms["BYTETracker"](args, frame_rate=int(frame_rate))

    def update(
        self,
        dets_xyxysc: np.ndarray,
        img_hw: Tuple[int, int],
        curr_img: Optional[np.ndarray] = None,
    ) -> List[Row]:
        """One step. ``dets_xyxysc``: ``(N, 5)`` float32 ``[x1, y1, x2, y2, score]``.

        ``img_info == img_size`` so the upstream box rescaling is the identity.
        The image is not used.
        """
        tracks = self._tracker.update(dets_xyxysc.copy(), img_hw, img_hw)
        rows: List[Row] = []
        for t in tracks:
            x, y, w, h = t.tlwh
            rows.append((int(t.track_id), float(x), float(y), float(w), float(h), float(t.score)))
        return rows
