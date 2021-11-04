import os
import cv2
import h5py
import pickle
import numpy as np

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



def bf_match(kp1, de1, kp2, de2,distance):
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

    # img_matches = None
    # img_matches = cv2.drawKeypoints(drone_img, kp1, img_matches)
    # cv2.imshow("img",img_matches)
    # cv2.waitKey(10000)

    pcount = len(good_matches)
    # if pcount < 100:
    #     print("Don't find enough match points")
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
    #
    # print("RANSAC", len(RR_matches))
    # img_RR_matches = cv2.drawMatches(drone_img, RR_KP1, satellite_img, RR_KP2, RR_matches, None,)
    # cv2.imshow("img",img_RR_matches)
    # # cv2.imwrite("sift_img.jpg", img_RR_matches)
    # cv2.waitKey()
    return pcount

# cv2.findFundamentalMat()


if __name__ == '__main__':
    drone_img = cv2.imread("img_dir" + os.sep + "150-1.jpg")
    drone_img = cv2.resize(drone_img, [512, 512])
    drone_img = cv2.GaussianBlur(drone_img, ksize=(3, 3), sigmaX=0.5)

    satellite_img = cv2.imread("img_dir" + os.sep + "s0.png")
    satellite_img = cv2.resize(satellite_img, [512, 512])
    # satellite_img = cv2.GaussianBlur(satellite_img, ksize=(3, 3), sigmaX=1)
    
    # drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)
    # satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2GRAY)

    ## SIFT
    # SIFT = cv2.xfeatures2d.SIFT_create()
    # key1, des1 = SIFT.detectAndCompute(drone_img, None)
    # key2, des2 = SIFT.detectAndCompute(satellite_img, None)
    # count = bf_match(key1, des1, key2, des2, cv2.NORM_L2)
    # print(count)

    ## STAR and BRIEF
    # STAR = cv2.xfeatures2d.StarDetector_create()
    # BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    #
    # kp1 = STAR.detect(drone_img, None)
    # kp2 = STAR.detect(satellite_img, None)
    # # print(len(kp1))
    #
    # kp1, des1 = BRIEF.compute(drone_img, kp1)
    # kp2, des2 = BRIEF.compute(satellite_img, kp2)
    gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    print(dst, dst.shape)
    drone_img[dst > 0.05 * dst.max()] = [0, 0, 255]
    cv2.imshow('dst', drone_img)
    cv2.waitKey(0)




    temp = [kp1[0].pt, kp1[0].size, kp1[0].angle, kp1[0].response, kp1[0].octave,
            kp1[0].class_id, des1]

    f = open("save.txt", "wb")
    pickle.dump(temp, f, 0)
    f.close()
    f = open("save.txt", "rb")
    point = pickle.load(f)
    feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],size=point[1], angle=point[2],
                            response=point[3], octave=point[4], class_id=point[5])

    # descriptor = point[6]
    # print(kp1[0], des1)
    # print(feature, descriptor)
    count = bf_match(kp1, des1, kp2, des2,cv2.NORM_HAMMING)
    print(count)

    # print(temp)


    # f = h5py.File("test.h5", "w")
    # test = f.create_group('test')
    # print(temp)
    # data = np.array(temp[0])
    # # print(data)
    # test.create_dataset('pos', data)
    # print(des1.shape)
    # f['test'].attrs['size'] = temp[1]
    # f['test'].attrs['angle'] = temp[2]
    # f['test'].attrs['response'] = temp[3]
    # f['test'].attrs['octave'] = temp[4]
    # f['test'].attrs['class_id'] = temp[4]
    # test.create_dataset('des', data=np.random.rand(338,32))
    # f.close()
    #
    # f = h5py.File("test.h5", "r")
    # print(f)
    # print(f['kp1-0'])
    # print(f['test'].attrs['des'])








