import os
import re
import cv2
import glob
import sys
import pandas as pd
# sys.path.append("..")
from utils import get_yaml_value
from match import fl_match, bf_match

datasets_path = "../../Datasets"
height_list = ['150', '200', '250', '300']

Detector = None
Extractor = None

if get_yaml_value("detector") == "SIFT" and get_yaml_value("extractor") == "SIFT":
    Detector = cv2.SIFT_create()
    Extractor = cv2.SIFT_create()
elif get_yaml_value("detector") == "STAR" and get_yaml_value("extractor") == "BRIEF":
    Detector = cv2.xfeatures2d.StarDetector_create()
    Extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # detector = cv2.SIFT

match_dict = {}

for height in height_list:

    test_path = os.path.join(datasets_path, "Testing",height)
    # train_path = os.path.join(train_path, "300")
    # print(test_path)
    test_query_drone_path = os.path.join(test_path, "query_drone")
    test_query_satellite_path = os.path.join(test_path, "query_satellite")

    test_gallery_drone_path = os.path.join(test_path, "gallery_drone")
    test_gallery_satellite_path = os.path.join(test_path, "gallery_satellite")

    query_drone_list = glob.glob(os.path.join(test_query_drone_path, "*"))

    query_satellite_list = glob.glob(os.path.join(test_query_satellite_path, "*"))

    gallery_drone_list = glob.glob(os.path.join(test_gallery_drone_path, "*"))

    gallery_satellite_list = glob.glob(os.path.join(test_gallery_satellite_path, "*"))


    drone_list = sorted(query_drone_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
    satellite_list = sorted(query_satellite_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))

    gallery_drone_list = sorted(gallery_drone_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
    gallery_satellite_list = sorted(gallery_satellite_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))


    for num in range(len(query_satellite_list)):
        satellite_number = "{:0>4d}".format(num + 1)
        # print(satellite_number)
        satellite_img = glob.glob(os.path.join(satellite_list[int(satellite_number) - 1], "*"))[0]
        satellite_img = cv2.imread(satellite_img)
        kp1 = Detector.detect(satellite_img, None)
        kp1, des1 = Extractor.compute(satellite_img, kp1)
        # print(satellite_img)
        match_dict[satellite_number] = []
        # gallery_drone_list
        for i in gallery_drone_list:
            img_list = glob.glob(os.path.join(i, "*"))
            img_list = sorted(img_list, key=lambda x: int(re.findall("[0-9]+", x[-6:])[0]))
            sum_match_points = 0
            # print(img_list)
            for img in img_list:
                gallery_img = cv2.imread(img)
                gallery_img = cv2.resize(gallery_img, [512,512])
                kp2 = Detector.detect(gallery_img, None)
                kp2, des2 = Extractor.compute(gallery_img, kp2)
                match_points = bf_match(kp1, des1, kp2, des2)
                sum_match_points = match_points + sum_match_points
                match_dict[satellite_number].append(match_points)

            mean_match_points = sum_match_points / 50
            print("group: ", i, mean_match_points)
        df = pd.DataFrame(match_dict)
        df.to_csv("query_drone_%s.csv" % height)
        # break
