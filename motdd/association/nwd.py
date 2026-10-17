"""Normalized Wasserstein Distance (NWD) between bounding boxes.

Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object
Detection", arXiv:2110.13389, 2021.

A box ``(c_x, c_y, w, h)`` is modelled as a 2-D Gaussian with mean
``(c_x, c_y)`` and covariance ``diag((w/2)^2, (h/2)^2)``. For such Gaussians
the squared 2-Wasserstein distance has the closed form (paper Eq. 1)

    W2^2 = (c_x,a - c_x,b)^2 + (c_y,a - c_y,b)^2 + ((w_a - w_b)/2)^2 + ((h_a - h_b)/2)^2

and ``NWD = exp(-W2 / C)`` in ``(0, 1]``. ``C = 12.8`` px (the AI-TOD
default of Wang et al.).
"""
from __future__ import annotations

import numpy as np

DEFAULT_C = 12.8


def nwd_batch(boxes_a: np.ndarray, boxes_b: np.ndarray, C: float = DEFAULT_C) -> np.ndarray:
    """Pairwise NWD between ``(N, 4)`` and ``(M, 4)`` center-form boxes -> ``(N, M)``."""
    A = np.asarray(boxes_a, dtype=np.float64)
    B = np.asarray(boxes_b, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != 4:
        raise ValueError(f"boxes_a must be (N, 4); got {A.shape}")
    if B.ndim != 2 or B.shape[1] != 4:
        raise ValueError(f"boxes_b must be (M, 4); got {B.shape}")
    if C <= 0:
        raise ValueError(f"C must be > 0; got {C}")
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)

    dcx = A[:, None, 0] - B[None, :, 0]
    dcy = A[:, None, 1] - B[None, :, 1]
    dw = (A[:, None, 2] - B[None, :, 2]) / 2.0
    dh = (A[:, None, 3] - B[None, :, 3]) / 2.0
    w2_sq = dcx * dcx + dcy * dcy + dw * dw + dh * dh
    return np.exp(-np.sqrt(np.maximum(w2_sq, 0.0)) / C)


def tlbr_to_cxcywh(tlbr: np.ndarray) -> np.ndarray:
    """Convert ``(x1, y1, x2, y2)`` boxes to ``(c_x, c_y, w, h)``."""
    a = np.asarray(tlbr, dtype=np.float64)
    if a.ndim == 1:
        w = a[2] - a[0]
        h = a[3] - a[1]
        return np.array([a[0] + w / 2.0, a[1] + h / 2.0, w, h], dtype=np.float64)
    w = a[:, 2] - a[:, 0]
    h = a[:, 3] - a[:, 1]
    out = np.empty_like(a)
    out[:, 0] = a[:, 0] + w / 2.0
    out[:, 1] = a[:, 1] + h / 2.0
    out[:, 2] = w
    out[:, 3] = h
    return out
