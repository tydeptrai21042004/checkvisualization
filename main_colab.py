
import argparse
import numpy as np
import cv2
import torch

import sort
import homography_tracker
import utilities

# ---------------------------------------------------------
# ROI SETTINGS (in FULL-FRAME coordinates)
# Format: (x1, y1, x2, y2)
# Set to None to disable ROI for that camera.
# ---------------------------------------------------------
ROI1 = (500, 200, 1400, 900)   # example for cam1
ROI2 = None                    # e.g. (400, 150, 1500, 900) or None


def apply_roi(frame, roi):
    """
    Return cropped frame and (x_offset, y_offset).
    If roi is None, returns original frame and (0, 0).
    """
    if roi is None:
        return frame, (0, 0)

    x1, y1, x2, y2 = roi
    # Safety clamp
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    return frame[y1:y2, x1:x2], (x1, y1)


def main(opts):
    # -----------------------------
    # Open videos
    # -----------------------------
    video1 = cv2.VideoCapture(opts.video1)
    assert video1.isOpened(), f"Could not open video1 source {opts.video1}"
    video2 = cv2.VideoCapture(opts.video2)
    assert video2.isOpened(), f"Could not open video2 source {opts.video2}"

    # -----------------------------
    # Load homography (cam1 -> cam4)
    # -----------------------------
    cam4_H_cam1 = np.load(opts.homography)
    cam1_H_cam4 = np.linalg.inv(cam4_H_cam1)

    # List of homographies per camera (camera 0 is identity)
    homographies = [np.eye(3), cam1_H_cam4]

    # -----------------------------
    # Load YOLOv5 model (Ultralytics hub)
    # -----------------------------
    detector = torch.hub.load("ultralytics/yolov5", "yolov5m")
    detector.agnostic = True        # class-agnostic NMS
    detector.classes = [0]          # only "person"
    detector.conf = opts.conf       # confidence threshold

    # -----------------------------
    # Initialize SORT trackers (per camera)
    # -----------------------------
    trackers = [
        sort.Sort(
            max_age=opts.max_age,
            min_hits=opts.min_hits,
            iou_threshold=opts.iou_thres,
        )
        for _ in range(2)
    ]

    # Global multi-camera tracker
    global_tracker = homography_tracker.MultiCameraTracker(
        homographies, iou_thres=0.20
    )

    num_frames1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(num_frames1, num_frames2)

    # cam4 video is ~17 frames behind cam1 in the original code
    video2.set(cv2.CAP_PROP_POS_FRAMES, 17)

    # -----------------------------
    # Video writer (side-by-side)
    # -----------------------------
    fps = video1.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (w * 2, h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(opts.output, fourcc, fps, out_size)

    # -----------------------------
    # (Optional) prediction logs for MOT-style evaluation
    # preds_cam[i] = list of [frame, id, x, y, w, h, 1, -1, -1]
    # -----------------------------
    preds_cam = [[], []]  # 0 -> cam1, 1 -> cam4

    print(f"[INFO] Processing up to {min(num_frames, opts.max_frames)} frames...")

    for frame_idx in range(min(num_frames, opts.max_frames)):
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        if not (ret1 and ret2):
            break

        # Keep full frames for drawing / homography
        full_frames_bgr = [frame1.copy(), frame2.copy()]

        # -----------------------------
        # Apply ROI cropping for detection
        # -----------------------------
        frame1_roi, offset1 = apply_roi(frame1, ROI1)
        frame2_roi, offset2 = apply_roi(frame2, ROI2)

        # YOLO expects RGB
        frames_rgb = [frame1_roi[:, :, ::-1], frame2_roi[:, :, ::-1]]
        anno = detector(frames_rgb)

        dets_list = []
        tracks_list = []

        # -----------------------------
        # Per-camera detection & tracking
        # -----------------------------
        for cam_idx in range(len(anno)):
            det_raw = anno.xyxy[cam_idx].cpu().numpy()  # x1,y1,x2,y2,conf,cls

            if det_raw.size == 0:
                # No detections: still need to update tracker with empty
                empty_dets = np.empty((0, 5), dtype=float)
                empty_labels = np.empty((0,), dtype=int)
                trks = trackers[cam_idx].update(empty_dets, empty_labels)
                tracks_list.append(trks)
                dets_list.append(empty_dets)
                continue

            # Split YOLO output: [x1,y1,x2,y2,conf,cls]
            boxes = det_raw[:, :4].astype(np.float32)
            scores = det_raw[:, 4:5].astype(np.float32)  # (N,1)
            labels = det_raw[:, 5].astype(int)           # class ids

            # Map boxes from ROI coords back to FULL frame
            if cam_idx == 0:
                x_off, y_off = offset1
            else:
                x_off, y_off = offset2

            boxes[:, 0] += x_off  # x1
            boxes[:, 2] += x_off  # x2
            boxes[:, 1] += y_off  # y1
            boxes[:, 3] += y_off  # y2

            # Detections for SORT: [x1,y1,x2,y2,score]
            det_for_sort = np.hstack([boxes, scores])

            # Update SORT tracker (labels are class ids)
            trks = trackers[cam_idx].update(det_for_sort, labels)

            dets_list.append(det_for_sort)
            tracks_list.append(trks)

        # -----------------------------
        # Multi-camera global ID assignment
        # tracks_list: list of arrays [x1,y1,x2,y2,id,label] per camera
        # global_ids[cam_idx] is a dict: local_id -> global_id
        # -----------------------------
        global_ids = global_tracker.update(tracks_list)

        # -----------------------------
        # (Optional) log predictions for evaluation (MOT format)
        # -----------------------------
        if opts.save_mot:
            for cam_idx, trks in enumerate(tracks_list):
                if trks.size == 0:
                    continue
                id_map = global_ids[cam_idx]
                for t in trks:
                    x1, y1, x2, y2, local_id, label = t
                    global_id = id_map.get(int(local_id), int(local_id))
                    w_box = x2 - x1
                    h_box = y2 - y1
                    # MOTChallenge: frame, id, x, y, w, h, conf(=1), -1, -1
                    preds_cam[cam_idx].append(
                        [
                            frame_idx + 1,      # 1-based frame index
                            int(global_id),
                            float(x1),
                            float(y1),
                            float(w_box),
                            float(h_box),
                            1,
                            -1,
                            -1,
                        ]
                    )

        # -----------------------------
        # Draw tracks on FULL frames
        # -----------------------------
        vis_frames = []
        for cam_idx in range(2):
            vis = utilities.draw_tracks(
                full_frames_bgr[cam_idx],
                tracks_list[cam_idx],
                global_ids[cam_idx],
                cam_idx,
                classes=detector.names,
            )

            # (Optional) draw ROI rectangle for visualization
            roi = ROI1 if cam_idx == 0 else ROI2
            if roi is not None:
                x1, y1, x2, y2 = roi
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

            vis_frames.append(vis)

        # Side-by-side
        vis_concat = np.hstack(vis_frames)
        writer.write(vis_concat)

        if frame_idx % 50 == 0:
            print(f"[INFO] Processed frame {frame_idx}/{min(num_frames, opts.max_frames)}")

    writer.release()
    video1.release()
    video2.release()
    print(f"[INFO] Saved output video to {opts.output}")

    # -----------------------------
    # Save MOT-style prediction files (if requested)
    # -----------------------------
    if opts.save_mot:
        for cam_idx in range(2):
            out_path = f"{opts.pred_prefix}_cam{cam_idx+1}.txt"
            with open(out_path, "w") as f:
                for row in preds_cam[cam_idx]:
                    line = ",".join(str(v) for v in row)
                    f.write(line + "\n")
            print(f"[INFO] Saved MOT predictions for cam{cam_idx+1} to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", type=str, default="epfl/cam1.mp4")
    parser.add_argument("--video2", type=str, default="epfl/cam4.mp4")
    parser.add_argument("--homography", type=str, default="epfl/cam4_H_cam1.npy")
    parser.add_argument("--output", type=str, default="epfl/mc_mot_output.mp4")
    parser.add_argument("--iou-thres", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Limit frames to process for faster demo.",
    )
    parser.add_argument(
        "--save-mot",
        action="store_true",
        help="Save predictions in MOTChallenge txt format for evaluation.",
    )
    parser.add_argument(
        "--pred-prefix",
        type=str,
        default="epfl/pred",
        help="Prefix for MOT .txt files (one per camera).",
    )

    args = parser.parse_args()
    main(args)
