import numpy as np
import cv2
import argparse

def main(opts):
    # Open videos
    video1 = cv2.VideoCapture(opts.video1)
    assert video1.isOpened(), f"Could not open video1 source {opts.video1}"
    video2 = cv2.VideoCapture(opts.video2)
    assert video2.isOpened(), f"Could not open video2 source {opts.video2}"

    # Read first frame from each camera
    ok1, frame1 = video1.read()
    ok2, frame2 = video2.read()
    assert ok1 and ok2, "Could not read frames from videos"

    # SIFT feature detector
    feat_detector = cv2.SIFT_create(1000)

    kpts1, des1 = feat_detector.detectAndCompute(frame1, None)
    kpts2, des2 = feat_detector.detectAndCompute(frame2, None)

    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    cam4_H_cam1, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    assert cam4_H_cam1 is not None, "Homography estimation failed"

    np.save(opts.homography_pth, cam4_H_cam1)
    print(f"Saved homography to {opts.homography_pth}")

    video1.release()
    video2.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", type=str, required=True, help="Path to cam1.mp4")
    parser.add_argument("--video2", type=str, required=True, help="Path to cam4.mp4")
    parser.add_argument("--homography-pth", type=str, required=True,
                        help="Output .npy path for homography (e.g. epfl/cam4_H_cam1.npy)")
    opts = parser.parse_args()
    main(opts)
