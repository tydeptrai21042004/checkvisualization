# homography_tracker.py
import numpy as np


def modify_bbox_source(bboxes, homography):
    """
    Project bounding boxes from their camera into a common reference frame
    using a homography.

    Args:
        bboxes (np.ndarray): Array of shape (N, M) where the first 4 columns
            are [x0, y0, x1, y1] and the remaining columns are kept as-is
            (e.g. [track_id, label, ...]).
        homography (np.ndarray): 3x3 homography matrix.

    Returns:
        np.ndarray: Projected bounding boxes with the same shape as input.
    """
    bboxes = np.asarray(bboxes)

    # Ensure 2D shape: (N, M)
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(-1, bboxes.shape[0])

    if bboxes.shape[0] == 0:
        # No boxes; return empty array with correct shape
        return bboxes.copy()

    bboxes_ = []

    for bbox in bboxes:
        x0, y0, x1, y1, *keep = bbox

        p0 = homography @ np.array([x0, y0, 1.0], dtype=float)
        p1 = homography @ np.array([x1, y1, 1.0], dtype=float)

        # Homogeneous -> Cartesian
        if p0[2] == 0 or p1[2] == 0:
            # Degenerate projection, skip this bbox
            continue

        x0p = int(p0[0] / p0[2])
        y0p = int(p0[1] / p0[2])
        x1p = int(p1[0] / p1[2])
        y1p = int(p1[1] / p1[2])

        bboxes_.append([x0p, y0p, x1p, y1p] + list(keep))

    if len(bboxes_) == 0:
        # No valid projections; return empty with same number of columns
        return np.empty((0, bboxes.shape[1]), dtype=bboxes.dtype)

    return np.asarray(bboxes_, dtype=bboxes.dtype)


def iou_batch(bb_test, bb_gt):
    """
    Compute IoU between two sets of boxes.

    Args:
        bb_test: (N, 4) array of [x1, y1, x2, y2]
        bb_gt:   (M, 4) array of [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix.
    """
    if bb_test.size == 0 or bb_gt.size == 0:
        return np.zeros((bb_test.shape[0], bb_gt.shape[0]), dtype=float)

    bb_test = np.asarray(bb_test, dtype=float)
    bb_gt = np.asarray(bb_gt, dtype=float)

    # Expand dims for broadcast
    bb_test = np.expand_dims(bb_test, 1)  # (N, 1, 4)
    bb_gt = np.expand_dims(bb_gt, 0)      # (1, M, 4)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * \
        (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * \
        (bb_gt[..., 3] - bb_gt[..., 1])

    union = area_test + area_gt - inter
    # Avoid division by zero
    union = np.maximum(union, 1e-9)

    return inter / union


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Simple greedy association based on IoU between two sets of boxes.

    Args:
        detections: (D, >=4) array, first 4 cols [x1, y1, x2, y2]
        trackers:   (T, >=4) array, first 4 cols [x1, y1, x2, y2]
        iou_threshold: minimum IoU to consider a match.

    Returns:
        matches:              (K, 2) array of [det_idx, trk_idx]
        unmatched_detections: (D_unmatched,) array of detection indices
        unmatched_trackers:   (T_unmatched,) array of tracker indices
    """
    detections = np.asarray(detections)
    trackers = np.asarray(trackers)

    D = detections.shape[0]
    T = trackers.shape[0]

    if D == 0 or T == 0:
        matches = np.empty((0, 2), dtype=int)
        unmatched_detections = np.arange(D, dtype=int)
        unmatched_trackers = np.arange(T, dtype=int)
        return matches, unmatched_detections, unmatched_trackers

    iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])

    matched_indices = []

    # Greedy matching: pick max IOU, remove that row/col, repeat
    used_dets = set()
    used_trks = set()

    while True:
        if iou_matrix.size == 0:
            break

        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break

        d_idx, t_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)

        if d_idx in used_dets or t_idx in used_trks:
            # Safety
            iou_matrix[d_idx, t_idx] = -1.0
            continue

        matched_indices.append([d_idx, t_idx])
        used_dets.add(d_idx)
        used_trks.add(t_idx)

        # Invalidate this row and column
        iou_matrix[d_idx, :] = -1.0
        iou_matrix[:, t_idx] = -1.0

    if len(matched_indices) > 0:
        matched_indices = np.asarray(matched_indices, dtype=int)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = np.array(
        [d for d in range(D) if d not in used_dets],
        dtype=int,
    )
    unmatched_trackers = np.array(
        [t for t in range(T) if t not in used_trks],
        dtype=int,
    )

    return matched_indices, unmatched_detections, unmatched_trackers


class MultiCameraTracker:
    """
    Simple multi-camera fusion: uses homographies + IoU to decide which
    local track IDs from different cameras share the same global ID.
    """

    def __init__(self, homographies: list, iou_thres=0.2, max_age_global=3000):
        """
        Multi Camera Tracking class constructor.

        Args:
            homographies (list[np.ndarray]): list of 3x3 homographies,
                one per camera, mapping from that camera into a common
                reference frame (usually cam0).
            iou_thres (float): IOU threshold for matching between cameras.
            max_age_global (int): "age" threshold to drop very old local ids
                from the fusion dictionaries (prevents unbounded growth).
        """
        self.num_sources = len(homographies)
        self.homographies = homographies
        self.iou_thres = iou_thres
        self.next_id = 1
        self.max_age_global = max_age_global

        # For each camera:
        #   ids[i]:  local_id -> global_id
        #   age[i]:  local_id -> age (fusion-update steps since first seen)
        self.ids = [{} for _ in range(self.num_sources)]
        self.age = [{} for _ in range(self.num_sources)]

    def update(self, tracks: list):
        """
        Update global IDs given per-camera tracks.

        Args:
            tracks: list of length num_sources, each element is an array
                    of shape (Ni, >=5) where columns are at least:
                      [x1, y1, x2, y2, local_id, ...].

        Returns:
            list[dict]: self.ids – list of dicts mapping local_id -> global_id
                        for each camera.
        """
        # 1) Project tracks to common reference frame
        proj_tracks = []
        for i, trks in enumerate(tracks):
            proj_tracks.append(modify_bbox_source(trks, self.homographies[i]))

        # 2) For each pair of sources, match their projected boxes
        for i in range(self.num_sources):
            for j in range(i + 1, self.num_sources):
                matched_global = {}

                if proj_tracks[i].shape[0] == 0 and proj_tracks[j].shape[0] == 0:
                    continue

                matches, unmatches_i, unmatches_j = associate_detections_to_trackers(
                    proj_tracks[i], proj_tracks[j], iou_threshold=self.iou_thres
                )

                # --- Matched tracks ---
                for idx_i, idx_j in matches:
                    id_i = proj_tracks[i][idx_i][4]
                    id_j = proj_tracks[j][idx_j][4]

                    match_i = self.ids[i].get(id_i)
                    match_j = self.ids[j].get(id_j)

                    # If track i already has a global id and is at least as old as j
                    if (
                        match_i is not None
                        and self.age[i].get(id_i, 0)
                        >= self.age[j].get(id_j, 0)
                        and not matched_global.get(match_i, False)
                    ):
                        self.ids[j][id_j] = match_i
                        matched_global[match_i] = True

                    # Else if track j has a global id
                    elif match_j is not None and not matched_global.get(
                        match_j, False
                    ):
                        self.ids[i][id_i] = match_j
                        matched_global[match_j] = True

                    # Neither has a global id yet
                    else:
                        self.ids[i][id_i] = self.next_id
                        self.ids[j][id_j] = self.next_id
                        matched_global[self.next_id] = True
                        self.next_id += 1

                    # Increment ages
                    self.age[i][id_i] = self.age[i].get(id_i, 0) + 1
                    self.age[j][id_j] = self.age[j].get(id_j, 0) + 1

                # --- Unmatched tracks in camera i ---
                for idx_i in unmatches_i:
                    id_i = proj_tracks[i][idx_i][4]
                    match_i = self.ids[i].get(id_i)

                    if match_i is None or matched_global.get(match_i, False):
                        self.ids[i][id_i] = self.next_id
                        matched_global[self.next_id] = True
                        self.next_id += 1

                    self.age[i][id_i] = self.age[i].get(id_i, 0) + 1

                # --- Unmatched tracks in camera j ---
                for idx_j in unmatches_j:
                    id_j = proj_tracks[j][idx_j][4]
                    match_j = self.ids[j].get(id_j)

                    if match_j is None or matched_global.get(match_j, False):
                        self.ids[j][id_j] = self.next_id
                        matched_global[self.next_id] = True
                        self.next_id += 1

                    self.age[j][id_j] = self.age[j].get(id_j, 0) + 1

        # 3) Optional cleanup: remove very old local_ids so dicts don't grow forever
        for i in range(self.num_sources):
            remove_ids = [
                lid for lid, a in self.age[i].items() if a > self.max_age_global
            ]
            for lid in remove_ids:
                self.age[i].pop(lid, None)
                self.ids[i].pop(lid, None)

        return self.ids
