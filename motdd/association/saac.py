"""Scale-Adaptive Association Cost (SAAC), paper Eq. 2.

    c(tau, d) = alpha(s_tau) * (1 - IoU(tau, d)) + (1 - alpha(s_tau)) * (1 - NWD(tau, d))

with the track scale ``s_tau = sqrt(w_tau * h_tau)`` of the (warped) Kalman
prediction and

    alpha(s) = clip((s - 8) / 32, 0, 1)

i.e. pure NWD for ``s <= 8`` px, pure IoU for ``s >= 40`` px and a linear
interpolation in between. The weight is conditioned on the track (row), not
on the detection. The cost lies on the same [0, 1] scale as ``1 - IoU``, so
it replaces ByteTrack's IoU distance at association stages 1 and 2 without
changing any threshold.
"""
from __future__ import annotations

import numpy as np

from .nwd import DEFAULT_C, nwd_batch

# Knots of alpha(s) in pixels (fixed, see paper Sec. 3.2).
S_LO = 8.0
S_HI = 40.0


def saac_alpha(sizes: np.ndarray) -> np.ndarray:
    """``alpha(s) = clip((s - 8) / 32, 0, 1)`` for a 1-D array of track scales."""
    s = np.asarray(sizes, dtype=np.float64).reshape(-1)
    return np.clip((s - S_LO) / (S_HI - S_LO), 0.0, 1.0).astype(np.float64)


def iou_batch_cxcywh(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Axis-aligned IoU between ``(N, 4)`` and ``(M, 4)`` center-form boxes -> ``(N, M)``."""
    A = np.asarray(boxes_a, dtype=np.float64)
    B = np.asarray(boxes_b, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != 4:
        raise ValueError(f"boxes_a must be (N, 4); got {A.shape}")
    if B.ndim != 2 or B.shape[1] != 4:
        raise ValueError(f"boxes_b must be (M, 4); got {B.shape}")
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)

    a_x1 = A[:, 0] - A[:, 2] / 2.0
    a_y1 = A[:, 1] - A[:, 3] / 2.0
    a_x2 = A[:, 0] + A[:, 2] / 2.0
    a_y2 = A[:, 1] + A[:, 3] / 2.0

    b_x1 = B[:, 0] - B[:, 2] / 2.0
    b_y1 = B[:, 1] - B[:, 3] / 2.0
    b_x2 = B[:, 0] + B[:, 2] / 2.0
    b_y2 = B[:, 1] + B[:, 3] / 2.0

    inter_x1 = np.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = np.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = np.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = np.minimum(a_y2[:, None], b_y2[None, :])
    iw = np.maximum(0.0, inter_x2 - inter_x1)
    ih = np.maximum(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = np.maximum(0.0, a_x2 - a_x1) * np.maximum(0.0, a_y2 - a_y1)
    area_b = np.maximum(0.0, b_x2 - b_x1) * np.maximum(0.0, b_y2 - b_y1)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def saac_cost(pred_boxes: np.ndarray, det_boxes: np.ndarray, C: float = DEFAULT_C) -> np.ndarray:
    """SAAC cost matrix between track predictions (rows) and detections (columns).

    Parameters
    ----------
    pred_boxes : (N, 4) center-form boxes of the track predictions.
    det_boxes  : (M, 4) center-form boxes of the detections.
    C          : NWD normalization constant (12.8 px).

    Returns
    -------
    (N, M) float64 cost matrix with values in [0, 1].
    """
    P = np.asarray(pred_boxes, dtype=np.float64)
    D = np.asarray(det_boxes, dtype=np.float64)
    if P.ndim != 2 or D.ndim != 2 or P.shape[0] == 0 or D.shape[0] == 0:
        return np.zeros((P.shape[0] if P.ndim == 2 else 0,
                         D.shape[0] if D.ndim == 2 else 0), dtype=np.float64)

    iou = iou_batch_cxcywh(P, D)
    nwd = nwd_batch(P, D, C=C)
    sizes = np.sqrt(np.maximum(P[:, 2] * P[:, 3], 0.0))
    a = saac_alpha(sizes)[:, None]
    return a * (1.0 - iou) + (1.0 - a) * (1.0 - nwd)
