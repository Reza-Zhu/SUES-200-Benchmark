import glob
import os

import re
import time
import cv2
import pandas as pd
from root_sift import compute_root_sift
from feature_matcher import bf_match, fl_match
from Preprocessing import load_keypoint_descriptor

load_path = "/media/tp/data/save_dir"
heights = [150, 200, 250, 300]
for height in heights:

    sift_load_path = os.path.join(load_path, "sift", str(height))

    csv_path = "csv_dir/Rootsift_matches_%s" % str(height) + ".csv"

    query_path = os.path.join(sift_load_path, "query")
    gallery_path = os.path.join(sift_load_path, "gallery")

    query_list = glob.glob(os.path.join(query_path,"*"))
    gallery_list = glob.glob(os.path.join(gallery_path,"*"))

    query_list = sorted(query_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
    gallery_list = sorted(gallery_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))

    match_dict = {}

    for num in range(len(query_list)):
        since = time.time()
        query_number = "{:0>4d}".format(num + 1)
        query_txt = glob.glob(os.path.join(query_list[num], "*"))[0]
        kp1, des1 = load_keypoint_descriptor(query_txt)
        des1 = compute_root_sift(des1)
        match_dict[query_number] = []

        for j in gallery_list:
            # print(j)
            gallery_txt_list = glob.glob(os.path.join(j, "*"))
            gallery_txt_list = sorted(gallery_txt_list, key=lambda x: int(re.findall("[0-9]+", x[-6:])[0]))
            for txt in gallery_txt_list:
                kp2, des2 = load_keypoint_descriptor(txt)
                des2 = compute_root_sift(des2)
                match_numbers = bf_match(kp1, des1, kp2, des2, cv2.NORM_L2)
                match_dict[query_number].append(match_numbers)
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        df = pd.DataFrame(match_dict)
        df.to_csv(csv_path)
        # break