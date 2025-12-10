# main.py
import torch
import numpy as np
import cv2

import utilities
import homography_tracker

from pathlib import Path
from boxmot import DeepOCSORT  # BoxMOT tracker


def project_to_ref(homography, cx, cy):
    """
    Project an image point (cx, cy) from a camera into the reference plane
    (camera 0 coordinates) using a 3x3 homography.
    """
    p = homography @ np.array([cx, cy, 1.0], dtype=float)
    if p[2] == 0:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=float)


def classify_role(cam_idx, bbox, staff_zones):
    """
    Very simple rule-based classifier:
      - 'staff' if bottom-center of bbox is inside staff_zones[cam_idx] polygon
      - 'customer' otherwise

    bbox: [x1, y1, x2, y2]
    staff_zones: list of polygons (np.ndarray Nx2) or None per camera
    """
    if cam_idx >= len(staff_zones):
        return "customer"

    poly = staff_zones[cam_idx]
    if poly is None:
        return "customer"

    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2)
    cy = y2  # bottom-center ~ feet

    inside = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0
    return "staff" if inside else "customer"


def build_example_staff_zones(video1, video2):
    """
    Build example staff zones based on frame size.

    For each camera:
      - Staff zone = right 25% of the image, between 30% and 90% of height.
    Feel free to replace this with manually chosen polygons for your store.
    """
    caps = [video1, video2]
    staff_zones = []

    for cap in caps:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if w <= 0 or h <= 0:
            staff_zones.append(None)
            continue

        x1 = int(0.75 * w)
        x2 = w - 1
        y1 = int(0.30 * h)
        y2 = int(0.90 * h)

        poly = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        staff_zones.append(poly)

    return staff_zones


def main(opts):
    # -----------------------------
    # 1. Open videos
    # -----------------------------
    video1 = cv2.VideoCapture(opts.video1)
    assert video1.isOpened(), f"Could not open video1 source {opts.video1}"
    video2 = cv2.VideoCapture(opts.video2)
    assert video2.isOpened(), f"Could not open video2 source {opts.video2}"

    # -----------------------------
    # 2. Homographies
    # -----------------------------
    cam4_H_cam1 = np.load(opts.homography)
    cam1_H_cam4 = np.linalg.inv(cam4_H_cam1)

    homographies = [np.eye(3), cam1_H_cam4]

    # -----------------------------
    # 3. Example staff zones
    #    (edit this later with your real store layout)
    # -----------------------------
    staff_zones = build_example_staff_zones(video1, video2)
    # staff_zones[0] -> polygon (cam1), staff_zones[1] -> polygon (cam4) or None

    # -----------------------------
    # 4. Detector (YOLOv5s – person only, FP16)
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True  # speed up conv on fixed sizes

    # Using smaller model for real-time
    detector = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    detector = detector.to(device)
    detector.agnostic = True               # class-agnostic NMS

    detector.classes = [0]                 # only "person"
    detector.conf = opts.conf              # confidence threshold

    if device == "cuda":
        detector.half()                    # FP16 for speed on GPU

    # -----------------------------
    # 5. Per-camera trackers (BoxMOT DeepOCSORT)
    # -----------------------------
    reid_weights = Path(opts.reid_weights)

    trackers = [
        DeepOCSORT(
            model_weights=reid_weights,
            device=device,
            fp16=(device == "cuda"),  # half precision only on GPU
        )
        for _ in range(2)              # one tracker per camera
    ]

    # -----------------------------
    # 6. Multi-camera fusion
    # -----------------------------
    global_tracker = homography_tracker.MultiCameraTracker(
        homographies,
        iou_thres=opts.iou_thres,
    )

    # -----------------------------
    # 7. Frame count + cam sync
    # -----------------------------
    num_frames1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(num_frames1, num_frames2)

    # NOTE: Second video 'cam4.mp4' is 17 frames behind the first video 'cam1.mp4'
    video2.set(cv2.CAP_PROP_POS_FRAMES, 17)

    # -----------------------------
    # 8. Metrics for customer–staff interaction
    # -----------------------------
    all_customers = set()                    # set of global IDs that were ever customers
    customer_visible_frames = {}             # gid -> #frames where visible
    customer_interaction_frames = {}         # gid -> #frames with staff nearby
    frame_idx = 0

    # Optional: for "unique persons ever seen" (customers + staff)
    seen_global_ids = set()

    # -----------------------------
    # 9. Main loop
    # -----------------------------
    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        if not ret1 or not ret2:
            break

        frame_idx += 1
        if frame_idx > num_frames:
            break

        # BGR frames for tracker & drawing
        frames = [frame1, frame2]

        # RGB frames for YOLOv5 (and convert to float16 if needed)
        frames_rgb = []
        for f in frames:
            rgb = f[:, :, ::-1]  # BGR -> RGB
            if device == "cuda":
                # YOLOv5 will internally convert numpy to tensor and cast;
                # this conversion is mainly for consistency.
                frames_rgb.append(rgb)
            else:
                frames_rgb.append(rgb)

        # Run detector (on RGB) - batch of 2 images
        # detector returns a list-like of Results, one per image
        anno = detector(frames_rgb)

        tracks = []

        # -------------------------
        # Per-camera tracking
        # -------------------------
        for cam_idx in range(len(anno)):
            # YOLOv5 Results.xyxy is a tensor: Nx6 [x1, y1, x2, y2, conf, cls]
            det = anno.xyxy[cam_idx].detach().cpu().numpy()

            # Optional extra safety: filter to person class (0)
            if det.shape[0] > 0:
                person_mask = det[:, 5] == 0
                det = det[person_mask]

            if det.shape[0] == 0:
                # No detections for this camera → empty track array
                tracks.append(np.empty((0, 6), dtype=float))
                continue

            # BoxMOT expects float dets: (x1,y1,x2,y2,conf,cls)
            dets_for_tracker = det[:, :6].astype(float)

            # DeepOCSORT expects BGR image
            ts = trackers[cam_idx].update(dets_for_tracker, frames[cam_idx])

            if ts is None or ts.shape[0] == 0:
                tracks.append(np.empty((0, 6), dtype=float))
                continue

            # BoxMOT output: (x1,y1,x2,y2,track_id,conf,cls,ind)
            # Adapt to (x1,y1,x2,y2,track_id,label) for drawing + global tracker
            out = np.zeros((ts.shape[0], 6), dtype=float)
            out[:, 0:4] = ts[:, 0:4]               # bbox
            out[:, 4] = ts[:, 4]                   # local track id
            out[:, 5] = ts[:, 6]                   # class index (0 for person)
            tracks.append(out)

        # -------------------------
        # Multi-camera ID fusion
        # -------------------------
        global_ids = global_tracker.update(tracks)   # list of dicts: local_id -> global_id

        # -------------------------
        # Compute positions & roles per global ID (for metrics)
        # -------------------------
        positions_ref = {}      # gid -> 2D point in ref camera coordinates
        roles = {}              # gid -> "staff" or "customer"
        current_customers = set()
        current_staff = set()
        current_all = set()

        for cam_idx, trks in enumerate(tracks):
            if trks.size == 0:
                continue

            id_map = global_ids[cam_idx]  # dict: local_id -> global_id
            H = homographies[cam_idx]

            for row in trks:
                x1, y1, x2, y2, local_id, _ = row
                local_id = int(local_id)
                gid = int(id_map.get(local_id, local_id))

                current_all.add(gid)
                seen_global_ids.add(gid)

                # Project bottom-center into reference plane
                cx = 0.5 * (x1 + x2)
                cy = y2
                p_ref = project_to_ref(H, cx, cy)
                if p_ref is None:
                    continue

                positions_ref[gid] = p_ref

                role = classify_role(cam_idx, [x1, y1, x2, y2], staff_zones)
                roles[gid] = role

                if role == "staff":
                    current_staff.add(gid)
                else:
                    current_customers.add(gid)
                    all_customers.add(gid)

        # -------------------------
        # Update interaction metrics
        # -------------------------
        staff_positions = [
            positions_ref[g] for g in current_staff if g in positions_ref
        ]

        num_customers_now = len(current_customers)
        num_staff_now = len(current_staff)
        num_total_now = len(current_all)
        num_interacting_now = 0

        for gid in current_customers:
            if gid not in positions_ref:
                continue

            # Visible this frame
            customer_visible_frames[gid] = customer_visible_frames.get(gid, 0) + 1

            interacted = False
            if staff_positions:
                cpos = positions_ref[gid]
                dists = [np.linalg.norm(cpos - s) for s in staff_positions]
                if dists and min(dists) <= opts.interact_dist:
                    interacted = True

            if interacted:
                customer_interaction_frames[gid] = (
                    customer_interaction_frames.get(gid, 0) + 1
                )
                num_interacting_now += 1

        # Running global stats (for overlay)
        total_visible = sum(customer_visible_frames.values())
        total_interact = sum(
            customer_interaction_frames.get(gid, 0) for gid in all_customers
        )
        global_pct = (
            100.0 * total_interact / total_visible if total_visible > 0 else 0.0
        )

        # -------------------------
        # Draw tracks (per camera)
        # -------------------------
        for i in range(2):
            frames[i] = utilities.draw_tracks(
                frames[i],         # BGR
                tracks[i],
                global_ids[i],
                i,
                classes=detector.names,
            )

            # draw staff zone polygon (red) for visualization
            if staff_zones[i] is not None:
                poly = staff_zones[i].astype(int)
                cv2.polylines(frames[i], [poly], True, (0, 0, 255), 2)

        # Overlay stats on first camera
        text1 = (
            f"Frame {frame_idx}: total={num_total_now}, "
            f"customers={num_customers_now}, staff={num_staff_now}"
        )
        text2 = (
            f"Unique customers={len(all_customers)}, "
            f"interaction={global_pct:.1f}% "
            f"(interacting now={num_interacting_now})"
        )
        cv2.putText(
            frames[0],
            text1,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frames[0],
            text2,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        vis = np.hstack(frames)

        if not opts.no_display:
            cv2.namedWindow("Vis", cv2.WINDOW_NORMAL)
            cv2.imshow("Vis", vis)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    video1.release()
    video2.release()
    if not opts.no_display:
        cv2.destroyAllWindows()

    # -----------------------------
    # Final stats (printed -> works in Colab)
    # -----------------------------
    total_customers = len(all_customers)
    total_visible = sum(customer_visible_frames.values())
    total_interact = sum(
        customer_interaction_frames.get(gid, 0) for gid in all_customers
    )
    global_pct = 100.0 * total_interact / total_visible if total_visible > 0 else 0.0

    print("========== Customer–Staff Interaction Stats ==========")
    print(f"Total different customers (global IDs): {total_customers}")
    print(f"Total customer-visible frames: {total_visible}")
    print(f"Total customer-frames with staff nearby: {total_interact}")
    print(f"Global interaction time percentage: {global_pct:.2f}%")
    print(f"Total unique persons (customers + staff): {len(seen_global_ids)}")

    # Optional: per-customer breakdown
    for gid in sorted(all_customers):
        vis_f = customer_visible_frames.get(gid, 0)
        inter_f = customer_interaction_frames.get(gid, 0)
        pct = 100.0 * inter_f / vis_f if vis_f > 0 else 0.0
        print(
            f"Customer {gid}: visible {vis_f} frames, "
            f"interacting {inter_f} frames ({pct:.1f}%)"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", type=str, default="./epfl/cam1.mp4")
    parser.add_argument("--video2", type=str, default="./epfl/cam4.mp4")
    parser.add_argument("--homography", type=str, default="./cam4_H_cam1.npy")
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.3,
        help="IOU threshold to consider a match between two bounding boxes.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="(Unused with DeepOCSORT – kept for backward compatibility).",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="(Unused with DeepOCSORT – kept for backward compatibility).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence value for the YoloV5 detector.",
    )
    parser.add_argument(
        "--reid-weights",
        type=str,
        default="osnet_x0_25_msmt17.pt",
        help="Path to ReID weights for BoxMOT DeepOCSORT.",
    )
    parser.add_argument(
        "--interact-dist",
        type=float,
        default=80.0,
        help=(
            "Distance threshold (in reference-camera pixels) to consider "
            "a customer and staff member 'interacting'."
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV windows (useful for Google Colab / long runs).",
    )

    opts = parser.parse_args()
    main(opts)
