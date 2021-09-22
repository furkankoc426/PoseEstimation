import cv2
import numpy as np
from matplotlib import pyplot as plt


def calibrate_camera(vr3d, vr2d, image_size, K):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(vr3d, vr2d, image_size, K, np.zeros((14, 1)),
                                                       flags=(cv2.CALIB_USE_INTRINSIC_GUESS |
                                                              cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
                                                              cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                                                              cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
                                                              cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_ZERO_TANGENT_DIST |
                                                              cv2.CALIB_FIX_TAUX_TAUY))

    return mtx, dist, rvecs, tvecs


def extract_feature(img):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)
    return kp1, des1


def draw_features(img, kp, name):
    img_feature = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(), plt.imshow(img_feature), plt.title(name), plt.show()


def draw_matches(img1, kp1, img2, kp2, matches, name):
    img_matched = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, matchColor=(0, 255, 0),
                                     singlePointColor=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS |
                                                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(), plt.imshow(img_matched), plt.title(name), plt.show()


def match_features(kp1, des1, kp2, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE: int = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.3 * n.distance:
            good_matches.append(matches[i])

    return good_matches


def get_points_from_matches(matches, kp1, kp2):
    pt1 = [kp1[m.queryIdx].pt for (m, n) in matches]
    pt2 = [kp2[m.trainIdx].pt for (m, n) in matches]
    pt1 = np.float32(pt1).reshape(-1, 1, 2)
    pt2 = np.float32(pt2).reshape(-1, 1, 2)
    return pt1, pt2


def get_relative_pose(pt1, pt2, K):
    R = np.eye(3)
    t = np.zeros((3,1))
    x1 = np.vstack((pt1[0].reshape(2, 1), [1]))
    x2 = np.vstack((pt2[0].reshape(2, 1), [1]))

    E, _ = cv2.findEssentialMat(pt1, pt2, K)
    R1, R2, t1 = cv2.decomposeEssentialMat(E)

    M = np.matmul(R1.transpose(), t1)
    M = np.array([[0, -M[2], M[1]], [M[2], 0, -M[0]], [-M[1], M[0], 0]], dtype=object)
    X1 = np.matmul(M, np.matmul(np.linalg.inv(K), x1))
    X2 = np.matmul(M, np.matmul(np.linalg.inv(K), x2))

    if X1[2] * X2[2] > 0:
        R = R1
        t = t1
        if X1[2] < 0:
            t *= -1.0
        H = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
        return H

    M = np.matmul(R2.transpose(), t1)
    M = np.array([[0, -M[2], M[1]], [M[2], 0, -M[0]], [-M[1], M[0], 0]], dtype=object)
    X1 = np.matmul(M, np.matmul(np.linalg.inv(K), x1))
    X2 = np.matmul(M, np.matmul(np.linalg.inv(K), x2))

    if X1[2] * X2[2] > 0:
        R = R2
        t = t1
        if X1[2] < 0:
            t *= -1.0
        H = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
        return H

    H = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
    return H


def plot_trajectory(position1, position2, position3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = np.concatenate((position1[0:3, 3:4], position2[0:3, 3:4], position3[0:3, 3:4]),1)
    # daha iyi grafik elde edebilmek icin eklendi
    ax.plot(points[0], points[2], points[1])
    #ax.plot(points[0], points[1], points[2])
    plt.show()
    return

