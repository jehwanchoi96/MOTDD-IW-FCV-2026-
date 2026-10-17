#!/usr/bin/env python
"""Convert UAVDT-M (MOT part of UAVDT) into the MOTChallenge layout used here.

Input  (official release):
    <images>/<seq>/img000001.jpg ...                 (UAV-benchmark-M)
    <gt>/<seq>_gt.txt  or  <seq>_gt_whole.txt        (UAV-benchmark-MOTD GT)
        frame, target_id, x, y, w, h, out_of_view, occlusion, category

Output:
    <out>/<seq>/img1          -> symlink to <images>/<seq>
    <out>/<seq>/gt/gt.txt     frame, id, x, y, w, h, 1, category, visibility
    <out>/<seq>/seqinfo.ini

Rows are sorted by (frame, id). ``visibility`` is 1 when the box is not out
of view (out_of_view == 1); ``category`` is copied from the annotation
(-1 when the file does not provide it). With the evaluation settings of
``tools/eval_hota.py`` (``DO_PREPROC=False``) every row is a valid
ground-truth box; the class and visibility columns are informational.

Usage:
    python tools/prepare_uavdt_m.py --images /path/to/UAV-benchmark-M \
        --gt /path/to/UAV-benchmark-MOTD_v1.0/GT --out data/UAVDT-M --seqs M0203 M0205 ...
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


def convert_sequence(images: Path, gt_dir: Path, out: Path, seq: str) -> int:
    img_src = (images / seq).resolve()
    frames = sorted(p for p in img_src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not frames:
        raise FileNotFoundError(f"no frames in {img_src}")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"cannot read {frames[0]}")
    H, W = first.shape[:2]

    ann = gt_dir / f"{seq}_gt.txt"
    if not ann.exists():
        ann = gt_dir / f"{seq}_gt_whole.txt"
    rows = []
    with ann.open() as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frame, tid = int(p[0]), int(p[1])
            x, y, w, h = (float(v) for v in p[2:6])
            vis = 1 if len(p) > 6 and int(float(p[6])) == 1 else 0
            cat = int(float(p[8])) if len(p) > 8 else -1
            rows.append((frame, tid, f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{cat},{vis}"))
    rows.sort(key=lambda r: (r[0], r[1]))

    seq_dir = out / seq
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(r[2] for r in rows) + ("\n" if rows else ""))
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={seq}\n"
        "imDir=img1\n"
        "frameRate=30\n"
        f"seqLength={len(frames)}\n"
        f"imWidth={W}\n"
        f"imHeight={H}\n"
        "imExt=.jpg\n"
    )
    link = seq_dir / "img1"
    if not link.exists():
        os.symlink(img_src, link)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="UAV-benchmark-M root")
    ap.add_argument("--gt", type=Path, required=True, help="directory with <seq>_gt.txt files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    args = ap.parse_args()
    for seq in args.seqs:
        n = convert_sequence(args.images, args.gt, args.out, seq)
        print(f"{seq}: {n} boxes")


if __name__ == "__main__":
    main()
