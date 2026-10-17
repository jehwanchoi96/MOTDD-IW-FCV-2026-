#!/usr/bin/env python
"""Convert VisDrone2019-MOT (test-dev) into the MOTChallenge layout used here.

Input  (official release):
    <src>/sequences/<seq>/0000001.jpg ...
    <src>/annotations/<seq>.txt
        frame, target_id, x, y, w, h, score, category, truncation, occlusion

Output:
    <out>/<seq>/img1          -> symlink to <src>/sequences/<seq>
    <out>/<seq>/gt/gt.txt     frame, id, x, y, w, h, 1, category, visibility
    <out>/<seq>/seqinfo.ini

Rows of category 0 (ignored regions) and 11 (others) are dropped; the row
order of the annotation file is kept. ``visibility`` is 1 for boxes that are
neither truncated (truncation == 0) nor heavily occluded (occlusion <= 1),
and 0 otherwise. With the evaluation settings of ``tools/eval_hota.py``
(``DO_PREPROC=False``) every kept row is a valid ground-truth box; the
class and visibility columns are informational.

Usage:
    python tools/prepare_visdrone_mot.py --src /path/to/VisDrone2019-MOT-test-dev \
        --out data/VisDrone-MOT-test-dev
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

DROP_CATEGORIES = {"0", "11"}


def convert_sequence(src: Path, out: Path, seq: str) -> int:
    img_src = (src / "sequences" / seq).resolve()
    frames = sorted(p for p in img_src.iterdir() if p.suffix.lower() == ".jpg")
    if not frames:
        raise FileNotFoundError(f"no frames in {img_src}")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"cannot read {frames[0]}")
    H, W = first.shape[:2]

    rows = []
    with (src / "annotations" / f"{seq}.txt").open() as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 10 or p[7] in DROP_CATEGORIES:
                continue
            vis = 1 if (p[8] == "0" and p[9] in ("0", "1")) else 0
            rows.append(f"{p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]},1,{p[7]},{vis}\n")

    seq_dir = out / seq
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    (seq_dir / "gt" / "gt.txt").write_text("".join(rows))
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name = {seq}\n"
        "imdir = img1\n"
        "framerate = 30\n"
        f"seqlength = {len(frames)}\n"
        f"imwidth = {W}\n"
        f"imheight = {H}\n"
        "imext = .jpg\n\n"
    )
    link = seq_dir / "img1"
    if not link.exists():
        os.symlink(img_src, link)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="VisDrone2019-MOT-test-dev root")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    seqs = sorted(p.stem for p in (args.src / "annotations").glob("*.txt"))
    for seq in seqs:
        n = convert_sequence(args.src, args.out, seq)
        print(f"{seq}: {n} boxes")


if __name__ == "__main__":
    main()
