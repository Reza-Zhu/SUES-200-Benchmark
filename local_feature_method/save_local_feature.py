import glob
import os
import re
import time

import cv2
import h5py
from feature_matcher import bf_match
from utils import create_dir
from Preprocessing import get_datasets_list
import pickle
import numpy as np


def save_keypoint_descriptor(kp_tuple, des, save_path):
    data_dict = {}
    key_points = []
    for kp in kp_tuple:
        temp_kp = [kp.pt, kp.size, kp.angle,
                   kp.response, kp.octave,
                   kp.class_id]
        key_points.append(temp_kp)
    data_dict['keypoint'] = key_points
    data_dict['descriptor'] = des
    with open(os.path.join(save_path), "wb") as f:
        pickle.dump(data_dict, f, 0)
        f.close()

    # f = open(os.path.join(save_path, "0001.txt"), "rb")
    # point = pickle.load(f)
    #
    # print(point)

def load_keypoint_descriptor(load_path):
    kp_tuple = []
    f = open(os.path.join(load_path, "0001.txt"), "rb")
    img = pickle.load(f)
    for kp in img['keypoint']:
        KeyPoint = cv2.KeyPoint(x=kp[0][0],y=kp[0][1],size=kp[1], angle=kp[2],
                            response=kp[3], octave=kp[4], class_id=kp[5])
        kp_tuple.append(KeyPoint)
    kp_tuple = tuple(kp_tuple)
    descriptor = img['descriptor']
    return kp_tuple, descriptor


Detector = cv2.SIFT_create()
Extractor = cv2.SIFT_create()
match_dict = {}

height = [150, 200, 250, 300]
for i in height:
    query_save_path = os.path.join("save_dir", str(i))
    gallery_save_path = os.path.join("save_dir", str(i))
    create_dir(query_save_path)
    create_dir(gallery_save_path)
    query_save_path = os.path.join(query_save_path, "query")
    gallery_save_path = os.path.join(gallery_save_path, "gallery")
    create_dir(query_save_path)
    create_dir(gallery_save_path)

    query_satellite_list = get_datasets_list(i, "query_satellite")
    gallery_drone_list = get_datasets_list(i, "gallery_drone")
    print(query_satellite_list)
    # query_save_path = os.path.join(save_dir_path, query_satellite_list[0][-4:])
    # print(query_save_path)
    # create_dir(query_save_path)

    for num in range(len(query_satellite_list)):
        satellite_number = "{:0>4d}".format(num + 1)
        satellite_img = glob.glob(os.path.join(query_satellite_list[int(satellite_number) - 1], "*"))[0]
        # print(satellite_img)
        query_save_txt = os.path.join(query_save_path, satellite_number)
        create_dir(query_save_txt)
        query_save_txt = os.path.join(query_save_txt, "0.txt")
        print(query_save_txt)
        satellite_img = cv2.imread(satellite_img)
        kp1 = Detector.detect(satellite_img, None)
        kp1, des1 = Extractor.compute(satellite_img, kp1)
        save_keypoint_descriptor(kp1, des1, query_save_txt)
        # break

        # match_dict[satellite_number] = []

    for num in range(len(gallery_drone_list)):
        # since = time.time()
        drone_number = "{:0>4d}".format(num + 1)
        gallery_save_path_ = os.path.join(gallery_save_path, drone_number)
        # print(gallery_save_path)
        create_dir(gallery_save_path_)
        img_list = glob.glob(os.path.join(gallery_drone_list[num], "*"))
        img_list = sorted(img_list, key=lambda x: int(re.findall("[0-9]+", x[-6:])[0]))
        sum_match_points = 0

        for img_num in range(len(img_list)):
            gallery_img = cv2.imread(img_list[img_num])
            gallery_img = cv2.resize(gallery_img, [512,512])
            gallery_img = cv2.GaussianBlur(gallery_img, ksize=(3, 3), sigmaX=0.5)
            gallery_save_txt = os.path.join(gallery_save_path_, str(img_num)+".txt")
            print(gallery_save_txt)
            kp2 = Detector.detect(gallery_img, None)
            kp2, des2 = Extractor.compute(gallery_img, kp2)
            save_keypoint_descriptor(kp2, des2, gallery_save_txt)

            # print(des)
            # match_points = bf_match(kp1, des, kp2, des2, cv2.NORM_L2)
            # print(match_points)
        #     break
        # break

