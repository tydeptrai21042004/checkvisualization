import cv2
import numpy as np

# Global dict: centroids[src][track_id] = list[(cx, cy), ...]
centroids = {}


def apply_homography(uv, H):
    """
    Apply homography H to an array of points uv of shape (N, 2).
    """
    uv = np.asarray(uv, dtype=float)
    uv_ = np.zeros_like(uv, dtype=float)

    for idx, (u, v) in enumerate(uv):
        uvs = H @ np.array([u, v, 1.0]).reshape(3, 1)
        u_, v_, s_ = uvs.reshape(-1)
        u_ /= s_
        v_ /= s_
        uv_[idx] = [u_, v_]

    return uv_


def apply_homography_xyxy(xyxy, H):
    """
    Apply homography H to an array of boxes of shape (N, 4): [x1, y1, x2, y2].
    """
    xyxy = np.asarray(xyxy, dtype=float)
    xyxy_ = np.zeros_like(xyxy, dtype=float)

    for idx, (x1, y1, x2, y2) in enumerate(xyxy):
        x1_, y1_, s1 = H @ np.array([x1, y1, 1.0]).reshape(3, 1)
        x1_ /= s1
        y1_ /= s1

        x2_, y2_, s2 = H @ np.array([x2, y2, 1.0]).reshape(3, 1)
        x2_ /= s2
        y2_ /= s2

        xyxy_[idx] = [x1_, y1_, x2_, y2_]

    return xyxy_


def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image given a list/array of (x1, y1, x2, y2).
    """
    for bbox in bounding_boxes:
        bbox = np.asarray(bbox, dtype=float)
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_matches(img1, kpts1, img2, kpts2, matches):
    """
    Visualize feature matches between two images.
    kpts1, kpts2: list/array of (x, y) coordinates.
    matches: list of cv2.DMatch or similar with .distance.
    """
    vis = np.hstack([img1, img2])
    if not matches:
        return vis

    max_dist_val = max(match.distance for match in matches)
    width2 = img2.shape[1]

    for (src, dst, match) in zip(kpts1, kpts2, matches):
        src_x, src_y = int(src[0]), int(src[1])
        dst_x, dst_y = int(dst[0]) + width2, int(dst[1])

        color = (0, int(255 * (match.distance / max_dist_val)), 0)
        vis = cv2.line(vis, (src_x, src_y), (dst_x, dst_y), color, 1)

    return vis


def color_from_id(id_):
    """
    Deterministic color for a given track id.
    """
    np.random.seed(int(id_))
    return np.random.randint(0, 255, size=3).tolist()


def draw_tracks(image, tracks, ids_dict, src, classes=None):
    """
    Draw bounding boxes and IDs for each track.

    tracks: ndarray of shape (N, 6) = [x1, y1, x2, y2, local_id, label]
    ids_dict: mapping local_id -> global_id
    src: camera index (used as key in global 'centroids')
    classes: optional list/array mapping label index -> class name
    """
    vis = np.array(image)
    if tracks.size == 0:
        return vis

    bboxes = tracks[:, :4]
    ids = tracks[:, 4].astype(int)
    labels = tracks[:, 5].astype(int)

    # Ensure centroids dict has this source
    centroids[src] = centroids.get(src, {})

    for i, box in enumerate(bboxes):
        local_id = ids[i]
        global_id = ids_dict[local_id]
        color = color_from_id(global_id)

        # Box coordinates
        box = np.asarray(box, dtype=float)
        x1, y1, x2, y2 = box.astype(int)

        # Track centroid history
        if centroids is None:
            # Should never happen with current code, but keep logic
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)
        else:
            MAX_TRAIL = 30  # last 30 points (~1.5s at 20 FPS)

            trail = centroids[src].get(global_id, [])
            trail.append(((x1 + x2) // 2, (y1 + y2) // 2))
            if len(trail) > MAX_TRAIL:
                trail.pop(0)

            centroids[src][global_id] = trail
            vis = draw_history(vis, box, trail, color)


        # Label text
        if classes is None:
            label_text = str(labels[i])
        else:
            # Safe guard if label index is out of range
            if 0 <= labels[i] < len(classes):
                label_text = classes[labels[i]]
            else:
                label_text = str(labels[i])

        text = f"{label_text} {global_id}"
        vis = cv2.putText(
            vis, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2
        )

    return vis


def draw_label(image, x, y, label, track_id, color):
    """
    Draw just a text label at (x, y).
    """
    vis = np.array(image)
    text = f"{label} {track_id}"
    vis = cv2.putText(
        vis, text, (int(x), int(y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2
    )
    return vis


def draw_history(image, box, centroid_list, color):
    """
    Draw a bounding box and its historical centroids (trace) on an image.

    box: [x1, y1, x2, y2]
    centroid_list: list of (cx, cy)
    """
    vis = np.array(image)

    # Box
    box = np.asarray(box, dtype=float)
    x1, y1, x2, y2 = box.astype(int)
    thickness = 2
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

    # History
    if len(centroid_list) == 0:
        return vis

    cent_arr = np.asarray(centroid_list, dtype=int)

    for i, centroid in enumerate(cent_arr):
        cx, cy = int(centroid[0]), int(centroid[1])
        if i == 0:
            cv2.circle(vis, (cx, cy), 2, color, thickness=-1)
        else:
            prev_cx, prev_cy = int(cent_arr[i - 1][0]), int(cent_arr[i - 1][1])
            cv2.line(vis, (prev_cx, prev_cy), (cx, cy), color, thickness=2)

    return vis
