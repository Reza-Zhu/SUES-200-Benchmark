import os
import cv2
import pickle
import numpy as np


def fl_match(kp1=None, de1=None, kp2=None, de2=None):
    index_params = dict(algorithm=1, trees=4)
    search_params = dict(checks=128)
    fl = cv2.FlannBasedMatcher(index_params, search_params)
    matches = fl.knnMatch(de1, de2, 2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    # print(matchesMask)
    # 按照Lowe的论文进行比率测试
    count = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchesMask[i] = [1, 0]
            count = count + 1

    return matches, matchesMask
    # return count

    # print(count)
    # draw_params = dict(matchesMask=matchesMask,
    #                    flags=0)

    # img = cv2.drawMatchesKnn(drone_img, kp1, satellite_img, kp2, matches, None, **draw_params)

    # # cv2.imshow("img",img)
    # cv2.imwrite("sift_img.jpg", img)
    #
    # cv2.waitKey()


def bf_match(kp1=None, de1=None, kp2=None, de2=None, distance=cv2.NORM_L2):
    bf = cv2.BFMatcher(distance, crossCheck=True)
    bf_matches = bf.match(de1, de2)

    max_distance = 0.0
    min_distance = 10000.0
    # print(matches[0].distance)
    for i in range(len(bf_matches)):
        if bf_matches[i].distance < min_distance:
            min_distance = bf_matches[i].distance
        if bf_matches[i].distance > max_distance:
            max_distance = bf_matches[i].distance

    # print("min: ", min_distance)
    # print("max: ", max_distance)

    good_matches = []
    for i in range(len(bf_matches)):
        if bf_matches[i].distance < 0.8 * max_distance:
            good_matches.append(bf_matches[i])
    # print(len(good_matches))

    #
    # # img_matches = None
    # # img_matches = cv2.drawKeypoints(drone_img, kp1, img_matches)
    # # cv2.imshow("img",img_matches)
    # # cv2.waitKey(10000)
    #
    # pcount = len(good_matches)
    # # print(pcount)
    # # if pcount < 100:
    # #     print("Don't find enough match points")
    # #
    # RAN_KP1 = []
    # RAN_KP2 = []
    # for i in range(len(good_matches)):
    #     # print(good_matches[i])
    #     RAN_KP1.append(kp1[good_matches[i].queryIdx])
    #     RAN_KP2.append(kp2[good_matches[i].trainIdx])
    #
    # p01 = []
    # p02 = []
    # for i in range(len(good_matches)):
    #     p01.append(RAN_KP1[i].pt)
    #     p02.append(RAN_KP2[i].pt)
    #
    # H, RansacStatus = cv2.findFundamentalMat(np.array(p01), np.array(p02), cv2.FM_RANSAC)
    # RR_KP1 = []
    # RR_KP2 = []
    # RR_matches = []
    # index = 0
    # # print(len(RansacStatus))
    # for i in range(len(good_matches)):
    #     if RansacStatus[i] != 0:
    #         RR_KP1.append(RAN_KP1[i])
    #         RR_KP2.append(RAN_KP2[i])
    #         good_matches[i].queryIdx = index
    #         good_matches[i].trainIdx = index
    #         RR_matches.append(good_matches[i])
    #         index = index + 1
    #
    # print("RANSAC", len(RR_matches))
    # img_RR_matches = cv2.drawMatches(drone_img, RR_KP1, satellite_img, RR_KP2, RR_matches, None,)
    # cv2.imshow("img",img_RR_matches)
    # # cv2.imwrite("sift_img.jpg", img_RR_matches)
    # cv2.waitKey()
    return len(good_matches)


# cv2.findFundamentalMat()









