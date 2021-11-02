import os
import re
import cv2
import glob
import time
import pandas as pd
from feature_matcher import fl_match, bf_match
from Preprocessing import get_datasets_list
datasets_path = "../../Datasets"
height = 150

Detector = cv2.xfeatures2d.StarDetector_create()
Extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
match_dict = {}

query_satellite_list = get_datasets_list(height, "query_satellite")
gallery_drone_list = get_datasets_list(height, "gallery_drone")

for num in range(len(query_satellite_list)):
    satellite_number = "{:0>4d}".format(num + 1)
    # print(satellite_number)
    satellite_img = glob.glob(os.path.join(query_satellite_list[int(satellite_number) - 1], "*"))[0]
    print(satellite_img)
    satellite_img = cv2.imread(satellite_img)
    kp1 = Detector.detect(satellite_img, None)
    kp1, des1 = Extractor.compute(satellite_img, kp1)
    # print(satellite_img)
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
    df.to_csv("csv_dir" + os.sep + "query_star_brief_%s.csv" % height)
