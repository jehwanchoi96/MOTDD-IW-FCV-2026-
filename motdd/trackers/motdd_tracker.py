"""MOTDD tracker: ByteTrack + CMC + SAAC + PGR.

Per frame (paper Fig. 3):

1. CMC.   ``H_t`` is estimated between frames ``t-1`` and ``t`` (ORB +
          MAGSAC++). After the Kalman prediction, the centre of every track
          prior (active, lost and unconfirmed) is warped by ``H_t``.
2. Stage 1. High-score detections (score > ``track_thresh``) are matched to
          the active and lost tracks with ByteTrack's score-fused cost and
          threshold ``match_thresh``. The geometric cost is ``1 - IoU`` or,
          with SAAC enabled, the SAAC cost (Eq. 2).
3. Stage 2. Low-score detections (0.1 < score < ``track_thresh``) are matched
          to the remaining active tracks (threshold 0.5), again with
          ``1 - IoU`` or SAAC. Active tracks left unmatched are marked lost.
4. Unconfirmed tracks are matched to the remaining high-score detections
          with the standard ByteTrack IoU stage (threshold 0.7).
5. PGR.   For up to ``K`` tracks that became unmatched in this frame (longest
          tracklets first), the detector is re-invoked on a square crop of
          side ``max(crop_scale * sqrt(area), min_crop)`` centred on the
          warped prediction. Recovered candidates that duplicate a detection
          already matched in this frame are removed; the survivors pass a
          buffered-IoU gate (both boxes dilated by ratio ``r``, IoU >=
          ``tau_gate``), and the best survivor (gate IoU x score) re-activates
          the track with its original identity.
6. New tracks are initialised from the unmatched high-score detections and
          the ByteTrack bookkeeping is applied.

With all three modules disabled the procedure is the ByteTrack update.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np

from .._third_party import load_bytetrack
from ..association.nwd import DEFAULT_C, tlbr_to_cxcywh
from ..association.saac import saac_cost
from ..cmc.homography import CameraMotionCompensator, warp_mean_position

Row = Tuple[int, float, float, float, float, float]


# ---------------------------------------------------------------- box helpers
def _iou_matrix_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between ``(N, 4)`` and ``(M, 4)`` xyxy arrays -> ``(N, M)``."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    lx = np.maximum(a[:, None, 0], b[None, :, 0])
    ly = np.maximum(a[:, None, 1], b[None, :, 1])
    rx = np.minimum(a[:, None, 2], b[None, :, 2])
    ry = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(rx - lx, 0, None)
    ih = np.clip(ry - ly, 0, None)
    inter = iw * ih
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def _dilate_xyxy(boxes: np.ndarray, ratio: float) -> np.ndarray:
    """Grow every box by ``ratio * w / 2`` and ``ratio * h / 2`` on each side."""
    if ratio <= 0 or boxes.size == 0:
        return boxes
    out = boxes.astype(np.float64).copy()
    w = out[:, 2] - out[:, 0]
    h = out[:, 3] - out[:, 1]
    dw = 0.5 * ratio * w
    dh = 0.5 * ratio * h
    out[:, 0] -= dw
    out[:, 1] -= dh
    out[:, 2] += dw
    out[:, 3] += dh
    return out


def buffered_iou(tracks_xyxy: np.ndarray, dets_xyxy: np.ndarray, ratio: float) -> np.ndarray:
    """Buffered IoU ``BIoU_r``: IoU after dilating both box sets by ``ratio``."""
    if tracks_xyxy.size == 0 or dets_xyxy.size == 0:
        return np.zeros((tracks_xyxy.shape[0], dets_xyxy.shape[0]), dtype=np.float64)
    return _iou_matrix_xyxy(_dilate_xyxy(tracks_xyxy, ratio), _dilate_xyxy(dets_xyxy, ratio))


def _tracks_to_cxcywh(tracks: List[Any]) -> np.ndarray:
    if not tracks:
        return np.zeros((0, 4), dtype=np.float64)
    tlbrs = np.stack([np.asarray(t.tlbr, dtype=np.float64) for t in tracks], axis=0)
    return tlbr_to_cxcywh(tlbrs)


# -------------------------------------------------------------------- tracker
class MOTDDTracker:
    """ByteTrack with switchable CMC, SAAC and PGR.

    Parameters
    ----------
    params : dict
        ByteTrack parameters ``track_thresh`` (0.5), ``track_buffer`` (30),
        ``match_thresh`` (0.8), ``mot20`` (False).
    frame_rate : int
        Sequence frame rate (ByteTrack track-buffer semantics).
    cmc : dict
        ``enable`` (bool), ``max_features`` (1000), ``min_inliers`` (20).
    saac : dict
        ``enable`` (bool), ``C`` (NWD constant, 12.8).
    pgr : dict
        ``enable`` (bool), ``K`` (4), ``crop_scale`` (8.0), ``min_crop`` (128),
        ``dedup_iou`` (0.5), ``double_count_iou`` (0.3), ``gate_iou`` (0.3),
        ``gate_buffer_ratio`` (0.2).
    detector : ZoomDetector, optional
        Required when PGR is enabled (``motdd.detection.ZoomDetector``).
    """

    def __init__(
        self,
        params: dict,
        frame_rate: int = 30,
        cmc: Optional[dict] = None,
        saac: Optional[dict] = None,
        pgr: Optional[dict] = None,
        detector: Any = None,
    ) -> None:
        cmc = dict(cmc or {})
        saac = dict(saac or {})
        pgr = dict(pgr or {})

        # CMC
        self.cmc_enable = bool(cmc.get("enable", False))
        self._cmc = (
            CameraMotionCompensator(
                max_features=int(cmc.get("max_features", 1000)),
                min_inliers=int(cmc.get("min_inliers", 20)),
            )
            if self.cmc_enable
            else None
        )

        # SAAC
        self.saac_enable = bool(saac.get("enable", False))
        self.saac_C = float(saac.get("C", DEFAULT_C))

        # PGR
        self.pgr_enable = bool(pgr.get("enable", False))
        self.pgr_K = int(pgr.get("K", 4))
        self.pgr_crop_scale = float(pgr.get("crop_scale", 8.0))
        self.pgr_min_crop = int(pgr.get("min_crop", 128))
        self.pgr_dedup_iou = float(pgr.get("dedup_iou", 0.5))
        self.pgr_double_count_iou = float(pgr.get("double_count_iou", 0.3))
        self.pgr_gate_iou = float(pgr.get("gate_iou", 0.3))
        self.pgr_gate_buffer_ratio = float(pgr.get("gate_buffer_ratio", 0.2))
        self.detector = detector
        if self.pgr_enable and self.detector is None:
            raise ValueError("PGR requires a detector (motdd.detection.ZoomDetector)")

        # ByteTrack
        syms = load_bytetrack()
        self._STrack = syms["STrack"]
        self._TrackState = syms["TrackState"]
        self._matching = syms["matching"]
        self._joint = syms["joint_stracks"]
        self._sub = syms["sub_stracks"]
        self._remove_dup = syms["remove_duplicate_stracks"]
        self._args = SimpleNamespace(
            track_thresh=float(params.get("track_thresh", 0.5)),
            track_buffer=int(params.get("track_buffer", 30)),
            match_thresh=float(params.get("match_thresh", 0.8)),
            mot20=bool(params.get("mot20", False)),
        )
        # The BYTETracker instance holds the track lists, frame counter,
        # Kalman filter and track-buffer parameters.
        self._tracker = syms["BYTETracker"](self._args, frame_rate=int(frame_rate))

    # ------------------------------------------------------------ helpers
    def _warp(self, stracks: List[Any], H: np.ndarray) -> None:
        for tr in stracks:
            if tr.mean is None:
                continue
            tr.mean = warp_mean_position(tr.mean, H)

    def _geometric_cost(self, tracks: List[Any], dets: List[Any]) -> np.ndarray:
        """``1 - IoU`` (ByteTrack) or the SAAC cost."""
        if self.saac_enable:
            return saac_cost(_tracks_to_cxcywh(tracks), _tracks_to_cxcywh(dets), C=self.saac_C)
        return self._matching.iou_distance(tracks, dets)

    @staticmethod
    def _xyxy(strack: Any) -> Tuple[float, float, float, float]:
        tlwh = strack.tlwh
        return (tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])

    # ------------------------------------------------------------- update
    def update(
        self,
        dets_xyxysc: np.ndarray,
        img_hw: Tuple[int, int],
        curr_img: Optional[np.ndarray] = None,
    ) -> List[Row]:
        """One tracker step.

        Parameters
        ----------
        dets_xyxysc : (N, 5) float32 ``[x1, y1, x2, y2, score]`` primary detections.
        img_hw : ``(H, W)`` of the frame.
        curr_img : BGR frame; required when CMC or PGR is enabled.

        Returns
        -------
        List of ``(track_id, x, y, w, h, score)`` for the activated tracks.
        """
        STrack = self._STrack
        TrackState = self._TrackState
        matching = self._matching
        args = self._args
        tracker = self._tracker

        H_t = self._cmc.step(curr_img) if self.cmc_enable else None

        tracker.frame_id += 1
        activated_stracks: List[Any] = []
        refind_stracks: List[Any] = []
        lost_stracks_this_frame: List[Any] = []
        removed_stracks_this_frame: List[Any] = []

        if dets_xyxysc.size == 0:
            dets_xyxysc = np.zeros((0, 5), dtype=np.float32)
        img_h, img_w = img_hw[0], img_hw[1]

        scores = dets_xyxysc[:, 4]
        bboxes = dets_xyxysc[:, :4]
        remain_inds = scores > args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        detections = (
            [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets, scores_keep)]
            if len(dets) > 0
            else []
        )

        unconfirmed: List[Any] = []
        tracked_stracks: List[Any] = []
        for track in tracker.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # ---- Kalman prediction + CMC warp of the priors ------------------
        strack_pool = self._joint(tracked_stracks, tracker.lost_stracks)
        STrack.multi_predict(strack_pool)
        if self.cmc_enable:
            self._warp(strack_pool, H_t)
            self._warp(unconfirmed, H_t)

        # Detections matched in this frame (xyxy), used by PGR de-duplication.
        matched_det_xyxys: List[Tuple[float, float, float, float]] = []

        # ---- Stage 1: high-score detections ------------------------------
        if strack_pool and detections:
            cost = self._geometric_cost(strack_pool, detections)
            if not args.mot20:
                cost = matching.fuse_score(cost, detections)
            m1, u_tr1, u_det1 = matching.linear_assignment(cost, thresh=args.match_thresh)
        else:
            m1 = np.empty((0, 2), dtype=int)
            u_tr1 = tuple(range(len(strack_pool)))
            u_det1 = tuple(range(len(detections)))

        for itracked, idet in m1:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, tracker.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, tracker.frame_id, new_id=False)
                refind_stracks.append(track)
            matched_det_xyxys.append(self._xyxy(det))

        # ---- Stage 2: low-score detections vs remaining active tracks ----
        detections_second = (
            [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_second, scores_second)]
            if len(dets_second) > 0
            else []
        )
        r_tracked_stracks = [
            strack_pool[i] for i in u_tr1 if strack_pool[i].state == TrackState.Tracked
        ]
        if r_tracked_stracks and detections_second:
            cost2 = self._geometric_cost(r_tracked_stracks, detections_second)
            m2, u_track2, _ = matching.linear_assignment(cost2, thresh=0.5)
        else:
            m2 = np.empty((0, 2), dtype=int)
            u_track2 = tuple(range(len(r_tracked_stracks)))

        for itracked, idet in m2:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, tracker.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, tracker.frame_id, new_id=False)
                refind_stracks.append(track)
            matched_det_xyxys.append(self._xyxy(det))
        for it in u_track2:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks_this_frame.append(track)

        # ---- Unconfirmed tracks (ByteTrack IoU stage) --------------------
        remaining_high_dets = [detections[i] for i in u_det1]
        if unconfirmed and remaining_high_dets:
            dists = matching.iou_distance(unconfirmed, remaining_high_dets)
            if not args.mot20:
                dists = matching.fuse_score(dists, remaining_high_dets)
            m3, u_unc, u_det3 = matching.linear_assignment(dists, thresh=0.7)
        else:
            m3 = np.empty((0, 2), dtype=int)
            u_unc = tuple(range(len(unconfirmed)))
            u_det3 = tuple(range(len(remaining_high_dets)))
        for itracked, idet in m3:
            unconfirmed[itracked].update(remaining_high_dets[idet], tracker.frame_id)
            activated_stracks.append(unconfirmed[itracked])
            matched_det_xyxys.append(self._xyxy(remaining_high_dets[idet]))
        for it in u_unc:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks_this_frame.append(track)

        # ---- PGR: prediction-gated recovery ------------------------------
        if self.pgr_enable and lost_stracks_this_frame:
            rescued = self._recover(
                lost_stracks_this_frame, curr_img, (img_h, img_w), matched_det_xyxys
            )
            refind_stracks.extend(rescued)
            rescued_ids = {id(t) for t in refind_stracks}
            lost_stracks_this_frame = [
                t for t in lost_stracks_this_frame if id(t) not in rescued_ids
            ]

        # ---- New tracks ---------------------------------------------------
        for inew in u_det3:
            track = remaining_high_dets[inew]
            if track.score < tracker.det_thresh:
                continue
            track.activate(tracker.kalman_filter, tracker.frame_id)
            activated_stracks.append(track)

        # ---- Remove lost tracks older than the track buffer --------------
        for track in tracker.lost_stracks:
            if tracker.frame_id - track.end_frame > tracker.max_time_lost:
                track.mark_removed()
                removed_stracks_this_frame.append(track)

        # ---- ByteTrack bookkeeping ---------------------------------------
        tracker.tracked_stracks = [
            t for t in tracker.tracked_stracks if t.state == TrackState.Tracked
        ]
        tracker.tracked_stracks = self._joint(tracker.tracked_stracks, activated_stracks)
        tracker.tracked_stracks = self._joint(tracker.tracked_stracks, refind_stracks)
        tracker.lost_stracks = self._sub(tracker.lost_stracks, tracker.tracked_stracks)
        tracker.lost_stracks.extend(lost_stracks_this_frame)
        tracker.lost_stracks = self._sub(tracker.lost_stracks, tracker.removed_stracks)
        tracker.removed_stracks.extend(removed_stracks_this_frame)
        tracker.tracked_stracks, tracker.lost_stracks = self._remove_dup(
            tracker.tracked_stracks, tracker.lost_stracks
        )

        rows: List[Row] = []
        for t in tracker.tracked_stracks:
            if not t.is_activated:
                continue
            x, y, w, h = t.tlwh
            rows.append((int(t.track_id), float(x), float(y), float(w), float(h), float(t.score)))
        return rows

    # ---------------------------------------------------------------- PGR
    def _recover(
        self,
        newly_lost: List[Any],
        curr_img: np.ndarray,
        img_hw: Tuple[int, int],
        matched_det_xyxys: List[Tuple[float, float, float, float]],
    ) -> List[Any]:
        """Prediction-gated recovery for tracks that became unmatched this frame.

        Returns the re-activated tracks. ``matched_det_xyxys`` is extended in
        place with the recovered boxes.
        """
        STrack = self._STrack
        frame_id = self._tracker.frame_id
        img_h, img_w = img_hw

        # Candidate budget: K tracks, longest tracklets first.
        cands = sorted(
            newly_lost,
            key=lambda t: (-int(getattr(t, "tracklet_len", 0)), int(getattr(t, "start_frame", 0))),
        )[: self.pgr_K]

        # Square crop around each warped prediction.
        centres: List[Tuple[float, float]] = []
        crop_sizes: List[int] = []
        cand_preds: List[np.ndarray] = []
        for tr in cands:
            tlwh = tr.tlwh
            x, y, w, h = float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3])
            s = max(1.0, float(np.sqrt(max(1e-6, w * h))))
            side = int(max(self.pgr_crop_scale * s, float(self.pgr_min_crop)))
            side = min(side, max(img_w, img_h))
            centres.append((x + w / 2.0, y + h / 2.0))
            crop_sizes.append(side)
            cand_preds.append(np.array([x, y, x + w, y + h], dtype=np.float64))

        crops, metas = self.detector.make_crops(curr_img, centres, crop_sizes)
        zoom_out = self.detector.infer_crops(crops, metas)

        # Remove candidates that duplicate detections already matched this frame.
        matched_arr = (
            np.asarray(matched_det_xyxys, dtype=np.float64)
            if matched_det_xyxys
            else np.zeros((0, 4))
        )
        per_cand_kept: List[np.ndarray] = []
        for dets_k in zoom_out:
            if dets_k.shape[0] == 0:
                per_cand_kept.append(np.zeros((0, 6), dtype=np.float32))
                continue
            if matched_arr.shape[0] > 0:
                max_iou = _iou_matrix_xyxy(dets_k[:, :4], matched_arr).max(axis=1)
            else:
                max_iou = np.zeros((dets_k.shape[0],), dtype=np.float64)
            keep = (max_iou < self.pgr_dedup_iou) & (max_iou < self.pgr_double_count_iou)
            per_cand_kept.append(dets_k[keep])

        # Buffered-IoU gate; each track only sees the candidates of its own crop.
        rescued: List[Any] = []
        for k, tr in enumerate(cands):
            dk = per_cand_kept[k]
            if dk.shape[0] == 0:
                continue
            biou = buffered_iou(
                cand_preds[k].reshape(1, 4),
                dk[:, :4].astype(np.float64),
                self.pgr_gate_buffer_ratio,
            ).ravel()
            over = biou >= self.pgr_gate_iou
            if not np.any(over):
                continue
            combo = biou * dk[:, 4].astype(np.float64)
            idx = int(np.argmax(np.where(over, combo, -np.inf)))
            x1, y1, x2, y2, sc, _cls = dk[idx].tolist()
            new_det = STrack(np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32), float(sc))
            tr.re_activate(new_det, frame_id, new_id=False)
            rescued.append(tr)
            matched_det_xyxys.append((x1, y1, x2, y2))
        return rescued
