#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
entry_eval.py

1) Run YOLOv8 + ByteTrack + virtual line to detect "entry" events in a video.
2) Compare detected events with ground-truth events (frames) and compute accuracy.

Usage examples:
---------------
# 1) Run detection
python entry_eval.py detect \
    --video store_entrance.mp4 \
    --model yolov8n.pt \
    --line-p1 100,400 \
    --line-p2 800,400 \
    --out detected_entries.csv \
    --device cuda

# 2) Evaluate accuracy (with ±5 frame tolerance)
python entry_eval.py eval \
    --det detected_entries.csv \
    --gt gt_entries.csv \
    --tol 5
"""

import argparse
import csv
import os
from typing import Dict, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# Geometry helpers
# =========================
def side_of_line(p: Tuple[float, float],
                 p1: Tuple[float, float],
                 p2: Tuple[float, float]) -> float:
    """
    Signed distance-ish (cross-product) to determine which side
    of directed segment p1 -> p2 the point p lies on.
    """
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


def parse_xy(s: str) -> Tuple[int, int]:
    """
    Parse 'x,y' into (x, y).
    """
    x_str, y_str = s.split(",")
    return int(x_str), int(y_str)


# =========================
# Detection: YOLOv8 + ByteTrack + line crossing
# =========================
def detect_entries(
    video_path: str,
    model_path: str,
    line_p1: Tuple[int, int],
    line_p2: Tuple[int, int],
    output_csv: str,
    conf: float = 0.4,
    device: str = "",
    save_vis: str = "",
    direction: str = "outside_to_inside",
):
    """
    Run YOLOv8 + ByteTrack on a single video and log entry events.

    - 'direction' is just used to interpret sign of crossing.
      By default we check from "negative side -> positive side",
      you can flip if needed.
    """
    # Load model
    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # set params for tracking
    # 'bytetrack.yaml' is bundled with ultralytics
    tracker_cfg = "bytetrack.yaml"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Track -> last_side map
    last_side: Dict[int, float] = {}
    # Track -> has_already_entered (avoid counting same person multiple times)
    already_entered: Dict[int, bool] = {}

    # Prepare CSV writer
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    csv_file = open(output_csv, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "time_sec", "track_id", "cx", "cy"])

    # Optional visualization
    if save_vis:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vis_writer = cv2.VideoWriter(save_vis, fourcc, fps, (w, h))
    else:
        vis_writer = None

    # Run YOLO tracking via ultralytics generator
    print(f"[INFO] Starting tracking with ByteTrack on {video_path}")
    frame_idx = 0
    stream = model.track(
        source=video_path,
        conf=conf,
        device=device if device else None,
        tracker=tracker_cfg,
        stream=True,
        classes=[0],  # only person
        verbose=False,
    )

    for result in stream:
        # ultralytics Results object
        boxes = result.boxes

        # Original frame for visualization
        frame = result.orig_img
        if frame is None:
            # Should not happen, but just in case
            ret, frame = cap.read()
            if not ret:
                break

        frame_idx += 1
        time_sec = frame_idx / fps

        # Draw line on frame
        cv2.line(frame, line_p1, line_p2, (0, 255, 255), 2)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # 'id' is ByteTrack ID, may be None at first
                track_id = box.id
                if track_id is None:
                    continue
                track_id = int(track_id)

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # bottom-center as person "foot"
                cx = 0.5 * (x1 + x2)
                cy = y2

                # current side
                cur_side = side_of_line((cx, cy), line_p1, line_p2)

                # Initialize last side if new track
                if track_id not in last_side:
                    last_side[track_id] = cur_side
                    already_entered[track_id] = False
                else:
                    prev_side = last_side[track_id]
                    last_side[track_id] = cur_side

                    # Skip if we already counted this track as entered
                    if already_entered[track_id]:
                        pass
                    else:
                        # Detect crossing: sign change
                        # Here we define: "outside" = negative side, "inside" = positive side,
                        # you may flip if your geometry is opposite.
                        if prev_side < 0 and cur_side > 0:
                            # CROSSING from outside -> inside
                            already_entered[track_id] = True
                            print(
                                f"[ENTRY] frame={frame_idx}, time={time_sec:.2f}s, id={track_id}"
                            )
                            writer.writerow(
                                [frame_idx, f"{time_sec:.3f}", track_id, f"{cx:.1f}", f"{cy:.1f}"]
                            )

                # draw bbox + id for visualization
                if vis_writer is not None:
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        # write vis frame
        if vis_writer is not None:
            vis_writer.write(frame)

        if total_frames and frame_idx % 50 == 0:
            print(f"[INFO] frame {frame_idx}/{total_frames}")

    csv_file.close()
    cap.release()
    if vis_writer is not None:
        vis_writer.release()

    print(f"[INFO] Detection finished. Events saved to {output_csv}")


# =========================
# Evaluation: compare detected vs GT
# =========================
def load_event_frames_from_csv(path: str, col_name: str = "frame") -> List[int]:
    frames: List[int] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if col_name not in reader.fieldnames:
            raise ValueError(
                f"CSV {path} does not contain required column '{col_name}'"
            )
        for row in reader:
            frames.append(int(row[col_name]))
    frames.sort()
    return frames


def match_events(
    gt_frames: List[int], det_frames: List[int], tolerance: int = 5
) -> Tuple[int, int, int]:
    """
    Greedy matching:
    - For each GT event frame, find closest unmatched detected event within ±tolerance frames.
    - If found => TP, else => FN.
    - Remaining detected events => FP.
    """
    used_det = [False] * len(det_frames)
    tp = 0
    fn = 0

    for gt in gt_frames:
        best_det_idx = -1
        best_dist = None
        for i, df in enumerate(det_frames):
            if used_det[i]:
                continue
            dist = abs(df - gt)
            if dist <= tolerance:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_det_idx = i
        if best_det_idx >= 0:
            # matched
            used_det[best_det_idx] = True
            tp += 1
        else:
            fn += 1

    fp = sum(1 for u in used_det if not u)
    return tp, fp, fn


def evaluate_events(det_csv: str, gt_csv: str, tolerance: int):
    det_frames = load_event_frames_from_csv(det_csv, col_name="frame")
    gt_frames = load_event_frames_from_csv(gt_csv, col_name="frame")

    print(f"[INFO] #GT events    : {len(gt_frames)}")
    print(f"[INFO] #Detected evt : {len(det_frames)}")

    tp, fp, fn = match_events(gt_frames, det_frames, tolerance=tolerance)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print("====== ENTRY EVENT EVALUATION ======")
    print(f"Tolerance (frames): {tolerance}")
    print(f"TP = {tp}, FP = {fp}, FN = {fn}")
    print(f"Precision = {prec:.3f}")
    print(f"Recall    = {rec:.3f}")
    print(f"F1-score  = {f1:.3f}")
    print("====================================")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Entry detection + evaluation")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # detect
    p_detect = subparsers.add_parser("detect", help="Run YOLO+ByteTrack and log entries")
    p_detect.add_argument("--video", required=True, type=str, help="Input video path")
    p_detect.add_argument("--model", default="yolov8n.pt", type=str, help="YOLO model")
    p_detect.add_argument(
        "--line-p1",
        required=True,
        type=str,
        help='Entrance line point 1 as "x,y" (image coords)',
    )
    p_detect.add_argument(
        "--line-p2",
        required=True,
        type=str,
        help='Entrance line point 2 as "x,y" (image coords)',
    )
    p_detect.add_argument(
        "--out",
        default="detected_entries.csv",
        type=str,
        help="Output CSV file for detected entry events",
    )
    p_detect.add_argument(
        "--conf", default=0.4, type=float, help="YOLO confidence threshold"
    )
    p_detect.add_argument(
        "--device",
        default="",
        type=str,
        help='Device string ("cuda", "cpu", "0", "0,1", ...). Empty = default.',
    )
    p_detect.add_argument(
        "--save-vis",
        default="",
        type=str,
        help="Optional path to save visualization video (with boxes & line)",
    )

    # eval
    p_eval = subparsers.add_parser(
        "eval", help="Evaluate detected entries vs ground-truth"
    )
    p_eval.add_argument(
        "--det", required=True, type=str, help="CSV with detected entries"
    )
    p_eval.add_argument(
        "--gt", required=True, type=str, help="CSV with ground-truth entries"
    )
    p_eval.add_argument(
        "--tol",
        default=5,
        type=int,
        help="Frame tolerance when matching events (e.g. ±5 frames)",
    )

    args = parser.parse_args()

    if args.mode == "detect":
        line_p1 = parse_xy(args.line_p1)
        line_p2 = parse_xy(args.line_p2)
        detect_entries(
            video_path=args.video,
            model_path=args.model,
            line_p1=line_p1,
            line_p2=line_p2,
            output_csv=args.out,
            conf=args.conf,
            device=args.device,
            save_vis=args.save_vis,
        )
    elif args.mode == "eval":
        evaluate_events(det_csv=args.det, gt_csv=args.gt, tolerance=args.tol)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
