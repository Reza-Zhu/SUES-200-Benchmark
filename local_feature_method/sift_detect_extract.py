import os
import re
import cv2
import glob
import time
import pandas as pd
from feature_matcher import fl_match, bf_match

datasets_path = "../../Datasets"
height = 300

Detector = cv2.SIFT_create()
Extractor = cv2.SIFT_create()

match_dict = {}
test_path = os.path.join(datasets_path, "Testing", str(height))

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
# print(len(satellite_list))
gallery_drone_list = sorted(gallery_drone_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
gallery_satellite_list = sorted(gallery_satellite_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))


for num in range(len(satellite_list)):
    satellite_number = "{:0>4d}".format(num + 1)
    satellite_img = glob.glob(os.path.join(satellite_list[int(satellite_number) - 1], "*"))[0]
    print(satellite_img)
    satellite_img = cv2.imread(satellite_img)
    kp1 = Detector.detect(satellite_img, None)
    kp1, des1 = Extractor.compute(satellite_img, kp1)
    match_dict[satellite_number] = []
    # gallery_drone_list
    for i in gallery_drone_list:
        since = time.time()
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
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        mean_match_points = sum_match_points / 50
        print("Img processing complete: ", i[-30:], "|| mean match pair number:", mean_match_points)
    df = pd.DataFrame(match_dict)
    df.to_csv("csv_dir" + os.sep + "query_sift_%s.csv" % height)
