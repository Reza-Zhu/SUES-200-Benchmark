import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def fl_match(kp1, de1, kp2, de2):
    fl = cv2.FlannBasedMatcher()
    matches = fl.knnMatch(de1, de2, 2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    # print(matchesMask)
    # 按照Lowe的论文进行比率测试
    count = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchesMask[i] = [1, 0]
            count = count + 1
    return count

    # print(count)
    # draw_params = dict(matchesMask=matchesMask,
    #                    flags=0)

    # img = cv2.drawMatchesKnn(drone_img, kp1, satellite_img, kp2, matches, None, **draw_params)
    # # cv2.imshow("img",img)
    # cv2.imwrite("sift_img.jpg", img)
    #
    # cv2.waitKey()



def bf_match(kp1, de1, kp2, de2):
    bf = cv2.BFMatcher(crossCheck=True)
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
    
    # img_matches = None
    # img_matches = cv2.drawKeypoints(drone_img, kp1, img_matches)
    # cv2.imshow("img",img_matches)
    # cv2.waitKey(10000)

    pcount = len(good_matches)
    # if pcount < 100:
    #     print("Don't find enough match points")
    return pcount
    #
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

    # # print("RANSAC", len(RR_matches))
    # img_RR_matches = cv2.drawMatches(drone_img, RR_KP1, satellite_img, RR_KP2, RR_matches, None)
    # cv2.imwrite("sift_img.jpg", img_RR_matches)
    # cv2.waitKey()

# cv2.findFundamentalMat()


if __name__ == '__main__':
    drone_img = cv2.imread("img_dir" + os.sep + "300-2.jpg")
    drone_img = cv2.resize(drone_img, [512, 512])

    satellite_img = cv2.imread("img_dir"+ os.sep +"s0.png")
    satellite_img = cv2.resize(satellite_img, [512, 512])

    # drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)
    # satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2GRAY)

    ## SIFT
    # SIFT = cv2.xfeatures2d.SIFT_create()
    # key1, des1 = SIFT.detectAndCompute(drone_img, None)
    # key2, des2 = SIFT.detectAndCompute(satellite_img, None)
    # bf_match(key1, des1, key2, des2)

    ## STAR and BRIEF
    STAR = cv2.xfeatures2d.StarDetector_create()
    BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    kp1 = STAR.detect(drone_img, None)
    kp2 = STAR.detect(satellite_img, None)
    kp1, des1 = BRIEF.compute(drone_img, kp1)
    kp2, des2 = BRIEF.compute(satellite_img, kp2)
    bf_match(kp1, des1, kp2, des2)
    # print(kp1,kp2)