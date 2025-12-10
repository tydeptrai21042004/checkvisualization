#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main_colab.py

Single-camera person detection + tracking + ROI-based counting.

- Detector: YOLOv5s (torch.hub, pretrained on COCO)
- Tracker:  SORT (sort.py in this repo)
- Only persons whose bottom-center lies inside a given ROI rectangle
  are tracked & counted.
- Output: on-screen overlay (if display enabled) + optional MP4 video.
"""

import os
import time
import argparse

import cv2
import numpy as np
import torch
from sort import Sort  # your existing tracker


# -----------------------
# Helpers
# -----------------------
def parse_roi(roi_str):
    """
    Parse ROI string "x1,y1,x2,y2" into ints.
    """
    try:
        x1, y1, x2, y2 = map(int, roi_str.split(","))
        return x1, y1, x2, y2
    except Exception as e:
        raise ValueError(f"Invalid ROI format '{roi_str}'. Use x1,y1,x2,y2") from e


def is_inside_roi(cx, cy, roi):
    """
    Check if a point (cx, cy) lies inside ROI = (x1,y1,x2,y2).
    """
    x1, y1, x2, y2 = roi
    return (cx >= x1) and (cx <= x2) and (cy >= y1) and (cy <= y2)


def color_from_id(id_):
    """
    Deterministic color for a given track id.
    """
    np.random.seed(int(id_ * 9973) & 0xFFFF)
    return tuple(int(c) for c in np.random.randint(0, 255, size=3))


# -----------------------
# Main
# -----------------------
def main(opts):
    # -----------------------------
    # 1. Open video
    # -----------------------------
    cap = cv2.VideoCapture(opts.video1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {opts.video1}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # -----------------------------
    # 2. ROI parsing
    # -----------------------------
    if opts.roi:
        roi = parse_roi(opts.roi)
    else:
        # If ROI is not specified, use full frame
        roi = (0, 0, width - 1, height - 1)

    print(f"[INFO] Using ROI rectangle: {roi}")

    # -----------------------------
    # 3. Detector (YOLOv5s – person only)
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True  # speed up CNN on constant input size

    print("[INFO] Loading YOLOv5s from torch.hub...")
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True, verbose=False
    ).to(device)

    model.conf = opts.conf
    model.agnostic = True
    model.classes = [0]  # only "person"

    if device == "cuda":
        model.half()  # FP16 on GPU

    # -----------------------------
    # 4. Tracker (SORT)
    # -----------------------------
    tracker = Sort()  # you already patched sort.py in the notebook

    # -----------------------------
    # 5. Output video writer (optional)
    # -----------------------------
    writer = None
    if opts.save_video:
        os.makedirs(os.path.dirname(opts.save_video), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            opts.save_video, fourcc, fps, (width, height)
        )
        print(f"[INFO] Saving output video to: {opts.save_video}")

    # -----------------------------
    # 6. Runtime stats
    # -----------------------------
    seen_ids = set()      # unique track IDs that ever entered ROI
    frame_idx = 0
    t0 = time.time()

    # -----------------------------
    # 7. Main loop
    # -----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Convert BGR -> RGB for YOLOv5
        img_rgb = frame[:, :, ::-1]

        # YOLO inference
        with torch.no_grad():
            results = model(img_rgb, size=opts.imgsz)
        det = results.xyxy[0].detach().cpu().numpy()  # Nx6 [x1,y1,x2,y2,conf,cls]

        # Filter: keep class 0 (person)
        if det.shape[0] > 0:
            person_mask = det[:, 5] == 0
            det = det[person_mask]

        # If no detections: update tracker with empty array
        if det.shape[0] == 0:
            det_for_sort = np.empty((0, 5), dtype=float)
        else:
            # -----------------------------
            # ROI gating before tracking
            # -----------------------------
            x1_roi, y1_roi, x2_roi, y2_roi = roi
            keep_idx = []
            for i, d in enumerate(det):
                x1, y1, x2, y2, conf, cls = d
                cx = 0.5 * (x1 + x2)
                cy = y2  # bottom-center
                if is_inside_roi(cx, cy, roi):
                    keep_idx.append(i)

            if len(keep_idx) == 0:
                det_for_sort = np.empty((0, 5), dtype=float)
            else:
                det = det[keep_idx]
                # SORT wants [x1, y1, x2, y2, score]
                det_for_sort = det[:, :5].astype(float)

        # -----------------------------
        # Update tracker
        # -----------------------------
        tracks = tracker.update(det_for_sort)  # (N, 5) = [x1,y1,x2,y2,track_id]

        current_ids = set()

        # Draw ROI
        cv2.rectangle(
            frame,
            (roi[0], roi[1]),
            (roi[2], roi[3]),
            (0, 0, 255),
            2,
        )

        # Draw tracks
        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)

            # bottom-center for check (just to be safe if tracker drifts)
            cx = 0.5 * (x1 + x2)
            cy = y2
            if not is_inside_roi(cx, cy, roi):
                # we only consider tracks inside ROI for counting & drawing
                continue

            current_ids.add(tid)
            seen_ids.add(tid)

            color = color_from_id(tid)
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
            cv2.putText(
                frame,
                f"ID {tid}",
                (x1i, y1i - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # -----------------------------
        # Overlays: counts + FPS
        # -----------------------------
        elapsed = time.time() - t0
        fps_est = frame_idx / elapsed if elapsed > 0 else 0.0

        text1 = f"Frame {frame_idx} | In ROI now: {len(current_ids)}"
        text2 = f"Unique persons in ROI: {len(seen_ids)}"
        text3 = f"Approx FPS: {fps_est:.1f}"

        cv2.putText(
            frame,
            text1,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text2,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text3,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Write output video if requested
        if writer is not None:
            writer.write(frame)

        # Optional display (disabled for Colab with --no-display)
        if not opts.no_display:
            cv2.imshow("Vis", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not opts.no_display:
        cv2.destroyAllWindows()

    print("============================================")
    print(f"Total frames processed: {frame_idx}")
    print(f"Total unique persons in ROI: {len(seen_ids)}")
    print("============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-camera person tracking with ROI (Colab-friendly)."
    )
    parser.add_argument(
        "--video1",
        type=str,
        required=True,
        help="Path to input video (e.g. test.mp4)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="YOLOv5 confidence threshold (default: 0.30)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size for YOLOv5 (default: 640)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="",
        help='ROI rectangle as "x1,y1,x2,y2". If empty, full frame is used.',
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default="",
        help="Optional path to save output MP4. If empty, no video is saved.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV display windows (use this in Colab).",
    )

    args = parser.parse_args()
    main(args)
