"""Re-invocation of the primary YOLOX-s detector on zoomed crops (PGR).

For every PGR candidate track a square crop centred on its warped prediction
is cut from the current frame, letterboxed to 640 x 640 (top-left placement,
pad value 114, the YOLOX convention) and passed through the same YOLOX-s
weights as the primary detection pool. Post-processing uses NMS IoU 0.65 and
discards candidates below confidence 0.1. Detections are mapped back to the
original image coordinates.

The module is inference-only; it never trains or modifies the detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch

from .._third_party import ensure_yolox_on_path

ensure_yolox_on_path()

from yolox.exp import get_exp  # type: ignore  # noqa: E402
from yolox.utils import fuse_model, postprocess  # type: ignore  # noqa: E402


@dataclass
class CropMeta:
    """Placement of a crop in the source image."""

    src_x1: int
    src_y1: int
    src_w: int
    src_h: int


def _letterbox(img: np.ndarray, tgt: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """Resize ``img`` into a ``tgt = (tH, tW)`` canvas (top-left, pad 114)."""
    H0, W0 = img.shape[:2]
    tH, tW = tgt
    r = min(tH / H0, tW / W0)
    newH, newW = int(round(H0 * r)), int(round(W0 * r))
    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((tH, tW, 3), 114, dtype=np.uint8)
    canvas[:newH, :newW] = resized
    return canvas, r


def _crop_source(img_bgr: np.ndarray, cx: float, cy: float, side: int) -> Tuple[np.ndarray, CropMeta]:
    """Square crop of side ``side`` centred at ``(cx, cy)``, clipped to the image."""
    H, W = img_bgr.shape[:2]
    half = side // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + side
    y2 = y1 + side
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(W, x2)
    y2c = min(H, y2)
    if x2c <= x1c or y2c <= y1c:
        # Crop entirely outside the image: 1x1 dummy, yields no detections.
        return np.zeros((1, 1, 3), dtype=np.uint8), CropMeta(x1c, y1c, 1, 1)
    crop = img_bgr[y1c:y2c, x1c:x2c].copy()
    return crop, CropMeta(x1c, y1c, x2c - x1c, y2c - y1c)


class ZoomDetector:
    """YOLOX detector wrapped for batched inference on per-track crops.

    Parameters
    ----------
    exp_file : path to the YOLOX ``Exp`` file describing the fine-tuned model.
    ckpt_path : path to the fine-tuned checkpoint.
    device : ``"cuda"``, ``"cuda:N"`` or ``"cpu"`` (CPU is used if CUDA is unavailable).
    input_size : letterbox target ``(H, W)``; 640 x 640 in the paper.
    conf_thres : confidence threshold for recovered candidates (0.1).
    nms_thres : NMS IoU (0.65, same as the primary detection pool).
    fp16 : half precision on CUDA.
    """

    def __init__(
        self,
        exp_file: Path,
        ckpt_path: Path,
        device: str = "cuda",
        input_size: Tuple[int, int] = (640, 640),
        conf_thres: float = 0.1,
        nms_thres: float = 0.65,
        fp16: bool = True,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = tuple(int(x) for x in input_size)
        self.conf_thres = float(conf_thres)
        self.nms_thres = float(nms_thres)
        self.fp16 = bool(fp16) and self.device.type == "cuda"

        exp = get_exp(str(exp_file), None)
        self.num_classes = int(exp.num_classes)
        model = exp.get_model()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=True)
        model = fuse_model(model)
        model.to(self.device).eval()
        if self.fp16:
            model.half()
        self.model = model

    def make_crops(
        self,
        img_bgr: np.ndarray,
        centres_xy: Sequence[Tuple[float, float]],
        crop_sizes: Sequence[int],
    ) -> Tuple[List[np.ndarray], List[CropMeta]]:
        """Cut one square crop per ``(centre, side)`` pair."""
        assert len(centres_xy) == len(crop_sizes)
        crops: List[np.ndarray] = []
        metas: List[CropMeta] = []
        for (cx, cy), side in zip(centres_xy, crop_sizes):
            crop, meta = _crop_source(img_bgr, cx, cy, int(side))
            crops.append(crop)
            metas.append(meta)
        return crops, metas

    @torch.no_grad()
    def infer_crops(self, crops: List[np.ndarray], metas: List[CropMeta]) -> List[np.ndarray]:
        """Detect on a batch of crops.

        Returns one ``(n_i, 6)`` float32 array per crop with columns
        ``[x1, y1, x2, y2, score, class_id]`` in original image coordinates.
        """
        if not crops:
            return []
        n = len(crops)
        tH, tW = self.input_size
        batch = np.zeros((n, 3, tH, tW), dtype=np.float32)
        scales = np.zeros(n, dtype=np.float32)
        for i, crop in enumerate(crops):
            canvas, r = _letterbox(crop, (tH, tW))
            batch[i] = canvas.astype(np.float32).transpose(2, 0, 1)
            scales[i] = r

        x = torch.from_numpy(batch).to(self.device)
        if self.fp16:
            x = x.half()
        out = self.model(x)
        results = postprocess(
            out.float(),
            num_classes=self.num_classes,
            conf_thre=self.conf_thres,
            nms_thre=self.nms_thres,
            class_agnostic=False,
        )

        out_arrays: List[np.ndarray] = []
        for i, det in enumerate(results):
            if det is None or det.numel() == 0:
                out_arrays.append(np.zeros((0, 6), dtype=np.float32))
                continue
            det_np = det.cpu().numpy()
            # YOLOX postprocess rows: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
            xyxy = det_np[:, :4].copy()
            xyxy /= scales[i]
            meta = metas[i]
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, meta.src_w) + meta.src_x1
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, meta.src_h) + meta.src_y1
            score = det_np[:, 4] * det_np[:, 5]
            cls_id = det_np[:, 6]
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            keep = (w > 1) & (h > 1) & (score >= self.conf_thres)
            out_np = np.concatenate([xyxy[keep], score[keep, None], cls_id[keep, None]], axis=1)
            out_arrays.append(out_np.astype(np.float32))
        return out_arrays
