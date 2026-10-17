#!/usr/bin/env python
"""Run one tracker configuration over a list of sequences.

Reads a configuration yaml from ``configs/`` and writes one MOTChallenge
tracker file per sequence:

    <out_root>/<benchmark>/<config name>/data/<seq>.txt
    (frame, id, x, y, w, h, score, -1, -1, -1)

Example
-------
    python tools/run_tracker.py --config configs/full_dynuav.yaml \
        --data-root data --det-root detections --weights-root weights \
        --device cuda

Paths in the yaml may use the placeholders ``{DATA_ROOT}``, ``{DET_ROOT}`` and
``{WEIGHTS_ROOT}``; relative paths are resolved against the repository root.
"""
from __future__ import annotations

# The association side runs on one CPU thread (deterministic and comparable
# across processes). These must be set before numpy / OpenCV / torch load.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import argparse  # noqa: E402
import configparser  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

try:
    cv2.setNumThreads(1)
except Exception:
    pass


# ------------------------------------------------------------------ helpers
def _resolve(path_tmpl: str, roots: Dict[str, str], **fmt) -> Path:
    s = str(path_tmpl)
    for key, val in roots.items():
        s = s.replace("{" + key + "}", val)
    if fmt:
        s = s.format(**fmt)
    p = Path(s)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _load_seqinfo(seqinfo_ini: Path) -> Tuple[int, int, int, str]:
    if not seqinfo_ini.exists():
        raise FileNotFoundError(f"missing {seqinfo_ini}")
    parser = configparser.ConfigParser()
    parser.read(seqinfo_ini)
    seq = parser["Sequence"]
    return (int(seq["seqLength"]), int(seq["imWidth"]), int(seq["imHeight"]),
            seq.get("imExt", ".jpg"))


def _read_det_txt(path: Path) -> Dict[int, np.ndarray]:
    """MOTChallenge detection file -> {frame: (N, 5) float32 [x1, y1, x2, y2, score]}."""
    if not path.exists():
        raise FileNotFoundError(f"missing detection file {path}")
    per_frame: Dict[int, list] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                fr = int(parts[0])
                x = float(parts[2]); y = float(parts[3])
                w = float(parts[4]); h = float(parts[5])
                conf = float(parts[6])
            except (ValueError, IndexError):
                continue
            per_frame.setdefault(fr, []).append([x, y, x + w, y + h, conf])
    return {fr: np.asarray(rows, dtype=np.float32) for fr, rows in per_frame.items()}


def _frame_layout(img_dir: Path, ext: str) -> Tuple[str, int, int]:
    """Infer ``<prefix><zero-padded index><ext>`` naming from the first frame."""
    candidates = sorted(img_dir.glob(f"*{ext}"))
    if not candidates:
        raise FileNotFoundError(f"no frames in {img_dir} with extension {ext}")
    stem = candidates[0].stem
    i = 0
    while i < len(stem) and not stem[i].isdigit():
        i += 1
    return stem[:i], len(stem) - i, int(stem[i:])


def _build_detector(cfg: dict, roots: Dict[str, str], device: str):
    from motdd.detection import ZoomDetector

    det = cfg["detector"]
    pgr = cfg["pgr"]
    return ZoomDetector(
        exp_file=_resolve(det["exp"], roots),
        ckpt_path=_resolve(det["weights"], roots),
        device=device,
        input_size=tuple(int(x) for x in pgr.get("input_size", [640, 640])),
        conf_thres=float(pgr.get("conf_thresh", 0.1)),
        nms_thres=float(pgr.get("nms_thresh", 0.65)),
        fp16=bool(pgr.get("fp16", True)),
    )


def _build_tracker(cfg: dict, detector):
    kind = cfg.get("tracker", "motdd")
    frame_rate = int(cfg.get("frame_rate", 30))
    if kind == "bytetrack":
        from motdd.trackers import ByteTrackBaseline

        return ByteTrackBaseline(cfg["bytetrack"], frame_rate=frame_rate)
    if kind == "motdd":
        from motdd.trackers import MOTDDTracker

        return MOTDDTracker(
            cfg["bytetrack"],
            frame_rate=frame_rate,
            cmc=cfg.get("cmc"),
            saac=cfg.get("saac"),
            pgr=cfg.get("pgr"),
            detector=detector,
        )
    raise ValueError(f"unknown tracker: {kind!r}")


def _sequences(cfg: dict, cli_seqs: Optional[List[str]]) -> List[str]:
    if cli_seqs:
        return list(cli_seqs)
    seqs = cfg["data"].get("sequences")
    if not seqs:
        raise ValueError("no sequences given (config 'data.sequences' or --seqs)")
    return [str(s) for s in seqs]


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seqs", nargs="*", default=None, help="subset of sequences (default: all in config)")
    ap.add_argument("--data-root", default="data", help="replaces {DATA_ROOT} (default: data)")
    ap.add_argument("--det-root", default="detections", help="replaces {DET_ROOT} (default: detections)")
    ap.add_argument("--weights-root", default="weights", help="replaces {WEIGHTS_ROOT} (default: weights)")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "outputs")
    ap.add_argument("--name", default=None, help="output tracker name (default: config 'name')")
    ap.add_argument("--device", default="cuda", help="device of the PGR detector (cuda | cpu)")
    args = ap.parse_args()

    with args.config.open("r") as f:
        cfg = yaml.safe_load(f)
    roots = {"DATA_ROOT": args.data_root, "DET_ROOT": args.det_root,
             "WEIGHTS_ROOT": args.weights_root}

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    cmc_on = bool((cfg.get("cmc") or {}).get("enable", False))
    pgr_on = bool((cfg.get("pgr") or {}).get("enable", False))
    needs_img = cmc_on or pgr_on

    detector = None
    if pgr_on:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        detector = _build_detector(cfg, roots, args.device)

    name = args.name or cfg["name"]
    bench = cfg["benchmark"]
    out_dir = args.out_root / bench / name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = _sequences(cfg, args.seqs)
    print(f"[{name}] benchmark={bench} sequences={len(seqs)} "
          f"cmc={cmc_on} saac={bool((cfg.get('saac') or {}).get('enable', False))} pgr={pgr_on}",
          flush=True)

    data = cfg["data"]
    total_frames, total_wall = 0, 0.0
    for i, seq in enumerate(seqs, 1):
        seq_dir = _resolve(data["seq_root"], roots) / seq
        seq_len, W, H, ext = _load_seqinfo(seq_dir / "seqinfo.ini")
        det_by_frame = _read_det_txt(_resolve(data["det_file"], roots, seq=seq))
        img_dir = seq_dir / "img1"
        if needs_img:
            prefix, digits, first = _frame_layout(img_dir, ext)

        # A fresh tracker per sequence (state and track IDs are reset).
        tracker = _build_tracker(cfg, detector)

        seen = set()
        t0 = time.time()
        with (out_dir / f"{seq}.txt").open("w") as f:
            for frame in range(1, seq_len + 1):
                dets = det_by_frame.get(frame, np.zeros((0, 5), dtype=np.float32))
                if dets.size == 0:
                    dets = np.zeros((0, 5), dtype=np.float32)
                img = None
                if needs_img:
                    img = cv2.imread(str(img_dir / f"{prefix}{frame - 1 + first:0{digits}d}{ext}"))
                    if img is None:
                        img = np.zeros((H, W, 3), dtype=np.uint8)
                for tid, x, y, w, h, sc in tracker.update(dets, (H, W), img):
                    if w <= 0 or h <= 0:
                        continue
                    seen.add(tid)
                    f.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{sc:.4f},-1,-1,-1\n")
        wall = time.time() - t0
        total_frames += seq_len
        total_wall += wall
        print(f"[{name}] [{i}/{len(seqs)}] {seq}: {len(seen)} ids, {seq_len} frames, "
              f"{1000.0 * wall / max(1, seq_len):.1f} ms/frame", flush=True)
    print(f"[{name}] done: {total_frames} frames, "
          f"{1000.0 * total_wall / max(1, total_frames):.1f} ms/frame -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
