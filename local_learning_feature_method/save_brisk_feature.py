import glob
import os
import re
import cv2
from utils import create_dir
from local_learning_feature_method.func import save_keypoint,get_datasets_list

match_dict = {}

Detector = cv2.BRISK_create(100)
# Extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

height = [150, 200, 250, 300]
root_dir = os.path.join("/media/tp/data/save_dir", "brisk")
create_dir(root_dir)


for i in height:
    query_save_path = os.path.join(root_dir, str(i))
    gallery_save_path = os.path.join(root_dir, str(i))
    create_dir(query_save_path)
    create_dir(gallery_save_path)
    query_save_path = os.path.join(query_save_path, "query")
    gallery_save_path = os.path.join(gallery_save_path, "gallery")
    create_dir(query_save_path)
    create_dir(gallery_save_path)

    query_satellite_list = get_datasets_list(i, "query_satellite")
    gallery_drone_list = get_datasets_list(i, "gallery_drone")
    print(query_satellite_list)

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
        # kp1, des1 = Extractor.compute(satellite_img, kp1)
        save_keypoint(kp1, query_save_txt)
        # break

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
            gallery_img = cv2.resize(gallery_img, [512, 512])
            gallery_img = cv2.GaussianBlur(gallery_img, ksize=(3, 3), sigmaX=0.5)
            gallery_save_txt = os.path.join(gallery_save_path_, str(img_num) + ".txt")
            print(gallery_save_txt)
            kp2 = Detector.detect(gallery_img, None)
            # kp2, des2 = Extractor.compute(gallery_img, kp2)
            save_keypoint(kp2, gallery_save_txt)
