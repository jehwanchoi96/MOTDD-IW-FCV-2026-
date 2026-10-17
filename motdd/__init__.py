"""MOTDD: training-free multi-object tracking for dynamic drone imagery.

Reference implementation of the IW-FCV 2026 paper
"Multi-Object Tracking for Dynamic Drone Imagery".

The tracker extends ByteTrack with three modules that can be switched on and
off independently (the eight configurations of the paper's ablation):

* CMC  - homography-based camera motion compensation (ORB + MAGSAC++) applied
         to the Kalman prior of every track (``motdd.cmc``).
* SAAC - Scale-Adaptive Association Cost, a track-scale-conditioned linear
         interpolation of IoU and NWD used at association stages 1 and 2
         (``motdd.association``).
* PGR  - Prediction-Gated Recovery, re-detection on a zoomed crop around the
         warped prediction of tracks that became unmatched in the current
         frame (``motdd.trackers.motdd_tracker`` + ``motdd.detection``).
"""

__version__ = "1.0.0"
