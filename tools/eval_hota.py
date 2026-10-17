#!/usr/bin/env python
"""Pooled HOTA / DetA / AssA / MOTA / IDF1 / IDSW with the official TrackEval.

TrackEval is not part of this repository; install it separately (see
README). This script only configures it. Settings used for the paper:

* ``MotChallenge2DBox`` dataset, benchmark ``MOT17`` format, ``DO_PREPROC=False``
  (no pedestrian/distractor filtering: every ground-truth row whose 7th
  column is non-zero is evaluated, tracker class columns are ignored),
* metrics ``HOTA`` (averaged over localization thresholds), ``CLEAR`` and
  ``Identity`` at IoU 0.5,
* scores pooled over all sequences (``COMBINED_SEQ``).

Example
-------
    python tools/eval_hota.py --config configs/dynuav/full.yaml \
        --gt-folder data/DynUAV --trackers-folder outputs/DynUAV
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

# TrackEval still references NumPy aliases removed in NumPy 1.24.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    for _name, _val in (("float", float), ("int", int), ("bool", bool), ("object", object)):
        if not hasattr(np, _name):
            setattr(np, _name, _val)

try:
    import trackeval  # noqa: E402
    from trackeval.datasets import MotChallenge2DBox  # noqa: E402
    from trackeval.metrics import CLEAR, HOTA, Identity  # noqa: E402
except ImportError:  # pragma: no cover
    sys.exit("TrackEval is required: see README (Evaluation).")


def evaluate(gt_folder: Path, trackers_folder: Path, tracker: str, seqs, out_folder: Path) -> dict:
    dataset_cfg = {
        "GT_FOLDER": str(gt_folder.resolve()),
        "TRACKERS_FOLDER": str(trackers_folder.resolve()),
        "TRACKERS_TO_EVAL": [tracker],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "SKIP_SPLIT_FOL": True,
        "SEQ_INFO": {s: None for s in seqs},  # lengths read from seqinfo.ini
        "DO_PREPROC": False,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_FOLDER": str(out_folder.resolve()),
        "OUTPUT_SUB_FOLDER": "",
        "PRINT_CONFIG": False,
    }
    eval_cfg = {
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "BREAK_ON_ERROR": True,
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_EMPTY_CLASSES": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    }
    metrics = [HOTA(), CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
               Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False})]
    res, _ = trackeval.Evaluator(eval_cfg).evaluate([MotChallenge2DBox(dataset_cfg)], metrics)
    combined = res[list(res.keys())[0]][tracker]["COMBINED_SEQ"]["pedestrian"]
    return {
        "HOTA": float(np.mean(combined["HOTA"]["HOTA"])),
        "DetA": float(np.mean(combined["HOTA"]["DetA"])),
        "AssA": float(np.mean(combined["HOTA"]["AssA"])),
        "MOTA": float(combined["CLEAR"]["MOTA"]),
        "IDF1": float(combined["Identity"]["IDF1"]),
        "IDSW": int(combined["CLEAR"]["IDSW"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-folder", type=Path, required=True,
                    help="benchmark root with <seq>/gt/gt.txt and <seq>/seqinfo.ini")
    ap.add_argument("--trackers-folder", type=Path, required=True,
                    help="folder containing <tracker>/data/<seq>.txt")
    ap.add_argument("--config", type=Path, default=None,
                    help="tracker config; provides the tracker name and sequence list")
    ap.add_argument("--tracker", default=None, help="tracker name (default: config 'name')")
    ap.add_argument("--seqs", nargs="*", default=None, help="sequence list (default: from config)")
    ap.add_argument("--out-folder", type=Path, default=None,
                    help="where TrackEval writes its files (default: <trackers-folder>/<tracker>)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text()) if args.config else {}
    tracker = args.tracker or cfg.get("name")
    seqs = args.seqs or (cfg.get("data") or {}).get("sequences")
    if not tracker or not seqs:
        sys.exit("need --tracker and --seqs, or --config")
    out_folder = args.out_folder or (args.trackers_folder / tracker)

    m = evaluate(args.gt_folder, args.trackers_folder, tracker, [str(s) for s in seqs], out_folder)
    print(f"{tracker}: HOTA {100 * m['HOTA']:.2f}  DetA {100 * m['DetA']:.2f}  "
          f"AssA {100 * m['AssA']:.2f}  MOTA {100 * m['MOTA']:.2f}  "
          f"IDF1 {100 * m['IDF1']:.2f}  IDSW {m['IDSW']}")


if __name__ == "__main__":
    main()
