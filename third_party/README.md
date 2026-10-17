# Third-party code

This repository does not copy any third-party source. Two upstream code
bases are cloned next to it, at the exact commits used for the paper:

| Project | License | Repository | Commit |
|---|---|---|---|
| ByteTrack | MIT | https://github.com/ifzhang/ByteTrack | `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` |
| YOLOX | Apache-2.0 | https://github.com/Megvii-BaseDetection/YOLOX | `6ddff4824372906469a7fae2dc3206c7aa4bbaee` |

```bash
git clone https://github.com/ifzhang/ByteTrack.git third_party/ByteTrack
git -C third_party/ByteTrack checkout d1bf0191adff59bc8fcfeaa0b33d3d1642552a99
git clone https://github.com/Megvii-BaseDetection/YOLOX.git third_party/YOLOX
git -C third_party/YOLOX checkout 6ddff4824372906469a7fae2dc3206c7aa4bbaee
```

What is used:

* **ByteTrack**: `yolox/tracker/` (Kalman filter, `STrack`, `BYTETracker`,
  matching helpers). The ByteTrack baseline is the unmodified `BYTETracker`;
  the MOTDD tracker re-implements the update step around the same building
  blocks.
* **YOLOX**: `yolox/exp`, `yolox/models`, `yolox/utils` to rebuild the
  fine-tuned YOLOX-s that PGR re-invokes on zoomed crops.

Both repositories contain a top-level package named `yolox`; `motdd`
handles the import order (`motdd/_third_party.py`). Neither needs to be
pip-installed. Other locations can be selected with the environment variables
`MOTDD_BYTETRACK_DIR` and `MOTDD_YOLOX_DIR`.

Evaluation uses the official TrackEval
(https://github.com/JonathonLuiten/TrackEval, commit
`12c8791b303e0a0b50f753af204249e622d0281a`), installed with pip; it is not
vendored either.
