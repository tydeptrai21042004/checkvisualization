import numpy as np
import cv2
import argparse
import sys


def estimate_homography(
    video1_path: str,
    video2_path: str,
    out_path: str,
    num_features: int = 1000,
    ratio: float = 0.75,
    ransac_thresh: float = 5.0,
):
    # Open videos (works fine in headless environments)
    cap1 = cv2.VideoCapture(video1_path)
    if not cap1.isOpened():
        raise RuntimeError(f"Could not open video1 source: {video1_path}")

    cap2 = cv2.VideoCapture(video2_path)
    if not cap2.isOpened():
        cap1.release()
        raise RuntimeError(f"Could not open video2 source: {video2_path}")

    # Read first frame from each camera
    ok1, frame1 = cap1.read()
    ok2, frame2 = cap2.read()
    cap1.release()
    cap2.release()

    if not ok1 or frame1 is None:
        raise RuntimeError("Could not read first frame from video1")
    if not ok2 or frame2 is None:
        raise RuntimeError("Could not read first frame from video2")

    # --------- Feature detector (SIFT with ORB fallback) ----------
    if hasattr(cv2, "SIFT_create"):
        print("[INFO] Using SIFT features", file=sys.stderr)
        feat_detector = cv2.SIFT_create(num_features)
        matcher_norm = cv2.NORM_L2
    else:
        # Colab sometimes lacks contrib build → use ORB
        print("[WARN] cv2.SIFT_create not available. Falling back to ORB.",
              file=sys.stderr)
        feat_detector = cv2.ORB_create(num_features)
        matcher_norm = cv2.NORM_HAMMING

    kpts1, des1 = feat_detector.detectAndCompute(frame1, None)
    kpts2, des2 = feat_detector.detectAndCompute(frame2, None)

    if des1 is None or des2 is None:
        raise RuntimeError(
            f"Could not compute descriptors: "
            f"des1={None if des1 is None else des1.shape}, "
            f"des2={None if des2 is None else des2.shape}"
        )

    # --------- KNN matching + Lowe's ratio test ----------
    bf = cv2.BFMatcher(matcher_norm, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m_n in matches:
        # Be robust in case some entries have <2 matches
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)

    print(f"[INFO] Total matches: {len(matches)}, good matches: {len(good)}",
          file=sys.stderr)

    if len(good) < 4:
        raise RuntimeError(
            f"Not enough good matches for homography: {len(good)} (< 4)"
        )

    src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # --------- RANSAC homography ----------
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise RuntimeError("Homography estimation failed")

    np.save(out_path, H)
    inliers = int(mask.sum()) if mask is not None else "N/A"
    print(f"[OK] Saved homography to {out_path}")
    print(f"[INFO] H:\n{H}")
    print(f"[INFO] Inliers: {inliers} / {len(good)}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate homography between first frames of two videos."
    )
    parser.add_argument("--video1", type=str, required=True,
                        help="Path to cam1.mp4 (source camera)")
    parser.add_argument("--video2", type=str, required=True,
                        help="Path to cam4.mp4 (target camera)")

    # Accept both --homography_pth and --homography-pth for convenience
    parser.add_argument(
        "--homography_pth", "--homography-pth",
        dest="homography_pth",
        type=str,
        required=True,
        help="Output .npy path for homography (e.g. epfl/cam4_H_cam1.npy)",
    )
    parser.add_argument("--num-features", type=int, default=1000,
                        help="Number of SIFT/ORB keypoints (default: 1000)")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="Lowe's ratio test threshold (default: 0.75)")
    parser.add_argument("--ransac-thresh", type=float, default=5.0,
                        help="RANSAC reprojection threshold in pixels (default: 5.0)")
    args = parser.parse_args()

    estimate_homography(
        video1_path=args.video1,
        video2_path=args.video2,
        out_path=args.homography_pth,
        num_features=args.num_features,
        ratio=args.ratio,
        ransac_thresh=args.ransac_thresh,
    )


if __name__ == "__main__":
    main()
