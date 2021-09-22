import numpy as np
import cv2
from matplotlib import pyplot as plt
import utils

np.set_printoptions(precision=3, suppress=True)


def main():
    # import 2d and 3d points
    vr2d = np.load('data/npy/vr2d.npy').reshape((1, -1, 2))
    vr3d = np.load('data/npy/vr3d.npy').reshape((1, -1, 3))

    # import images
    img1 = cv2.imread('data/png/img1.png', cv2.IMREAD_COLOR)
    img2 = cv2.imread('data/png/img2.png', cv2.IMREAD_COLOR)
    img3 = cv2.imread('data/png/img3.png', cv2.IMREAD_COLOR)

    # create initial calibration matrix
    cx = 960
    cy = 540
    f = 100
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)

    # calibrate camera with 2d-3d points
    K, dist, rvecs, tvecs = utils.calibrate_camera(vr3d, vr2d, img1.shape[0:2], K)

    print("Camera intrinsic parameters")
    print(K)

    # extract feature
    kp1, des1 = utils.extract_feature(img1)
    kp2, des2 = utils.extract_feature(img2)
    kp3, des3 = utils.extract_feature(img3)

    # match features
    matches12 = utils.match_features(kp1, des1, kp2, des2)
    matches13 = utils.match_features(kp1, des1, kp3, des3)
    matches23 = utils.match_features(kp2, des2, kp3, des3)

    # draw matches
    #utils.draw_matches(img1, kp1, img3, kp3, matches13, "img1 - img3 matches")
    #utils.draw_matches(img2, kp2, img3, kp3, matches23, "img2 - img3 matches")
    #utils.draw_matches(img1, kp1, img2, kp2, matches12, "img1 - img2 matches")

    # get match points and essential matrix
    pt12_1, pt12_2 = utils.get_points_from_matches(matches12, kp1, kp2)
    pt13_1, pt13_3 = utils.get_points_from_matches(matches13, kp1, kp3)
    pt23_2, pt23_3 = utils.get_points_from_matches(matches23, kp2, kp3)
    H12 = utils.get_relative_pose(pt12_1, pt12_2, K)
    H13 = utils.get_relative_pose(pt13_1, pt13_3, K)
    H23 = utils.get_relative_pose(pt23_2, pt23_3, K)

    print("Pose from 1 to 2")
    print(H12)
    print("Pose from 1 to 3")
    print(H13)
    print("Pose from 2 to 3")
    print(H23)

    position1 = np.eye(4)
    position12 = np.array(np.matmul(np.linalg.inv(H12), position1))
    position13 = np.array(np.matmul(np.linalg.inv(H13), position1))
    position123 = np.array(np.matmul(np.linalg.inv(H23), position12))

    print("6 DoF Position1")
    print(position1)
    print("6 DoF Position12")
    print(position12)
    print("6 DoF Position13")
    print(position13)
    print("6 DoF Position123")
    print(position123)

    utils.plot_trajectory(position1, position12, position13)


if __name__ == "__main__":
    main()
