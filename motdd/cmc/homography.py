"""Frame-to-frame homography estimation for camera motion compensation.

``H_t`` maps pixel coordinates of frame ``t-1`` onto frame ``t``. It is
estimated from ORB keypoints (1000 features) matched with a cross-checked
Hamming brute-force matcher and fitted with MAGSAC++ (``cv2.USAC_MAGSAC``,
reprojection threshold 3 px, 2000 iterations, confidence 0.999).

If fewer than ``min_inliers`` (20) inliers support the fit, the frame falls
back to the identity homography (no compensation).

The tracker applies ``H_t`` to the centre ``(c_x, c_y)`` of every Kalman
prior before association (paper Eq. 3); aspect ratio and height are left
unchanged.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def estimate_homography(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    *,
    max_features: int = 1000,
    ransac_reproj_threshold_px: float = 3.0,
    ransac_max_iters: int = 2000,
) -> Tuple[np.ndarray, int]:
    """Estimate ``H_t : image_{t-1} -> image_t``.

    Returns
    -------
    H : (3, 3) float64
        Homography, or the identity if the fit is not possible.
    n_inliers : int
        Number of MAGSAC++ inliers (0 if the fit failed).
    """
    eye = np.eye(3, dtype=np.float64)
    if prev_img.shape[:2] != curr_img.shape[:2]:
        raise ValueError(
            f"prev/curr frame shapes disagree: {prev_img.shape[:2]} vs {curr_img.shape[:2]}"
        )

    prev_gray = _to_gray(prev_img)
    curr_gray = _to_gray(curr_img)
    orb = cv2.ORB_create(nfeatures=int(max_features))
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return eye, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        return eye, 0

    pts_prev = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_curr = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        pts_prev,
        pts_curr,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=float(ransac_reproj_threshold_px),
        maxIters=int(ransac_max_iters),
        confidence=0.999,
    )
    if H is None or mask is None:
        return eye, 0

    n_inliers = int(mask.ravel().astype(bool).sum())
    if n_inliers < 4:
        return eye, n_inliers
    return H.astype(np.float64), n_inliers


class CameraMotionCompensator:
    """Stateful per-sequence CMC: keeps the previous frame and returns ``H_t``.

    Parameters
    ----------
    max_features : int
        ORB feature cap (1000).
    min_inliers : int
        Minimum MAGSAC++ inlier count; below it ``H_t`` is the identity (20).
    """

    def __init__(self, max_features: int = 1000, min_inliers: int = 20) -> None:
        self.max_features = int(max_features)
        self.min_inliers = int(min_inliers)
        self._prev_img: Optional[np.ndarray] = None

    def step(self, curr_img: np.ndarray) -> np.ndarray:
        """Return ``H_t`` for the new frame and remember it for the next call."""
        if self._prev_img is None:
            H = np.eye(3, dtype=np.float64)
        else:
            H, n_inliers = estimate_homography(
                self._prev_img, curr_img, max_features=self.max_features
            )
            if n_inliers < self.min_inliers:
                H = np.eye(3, dtype=np.float64)
        self._prev_img = curr_img
        return H


def warp_mean_position(mean: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp the centre ``(c_x, c_y)`` of an 8-d ByteTrack Kalman mean by ``H``."""
    pt = np.array([[[float(mean[0]), float(mean[1])]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, H).ravel()
    out = mean.copy()
    out[0], out[1] = float(warped[0]), float(warped[1])
    return out
