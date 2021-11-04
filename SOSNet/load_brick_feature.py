import glob
import re
import cv2
import torch
import sosnet_model
import tfeat_utils
import os
import time
from local_learning_feature_method.func import load_keypoint,get_datasets_list
from local_learning_feature_method.matcher import bf_match

load_path = "/media/tp/data/save_dir"
heights = [150, 200, 250, 300]
torch.no_grad()

# Init the 32x32 version of SOSNet
sosnet32 = sosnet_model.SOSNet32x32()
net_name = 'notredame'
sosnet32.load_state_dict(torch.load("sosnet-32x32-"+net_name+".pth"))
sosnet32.cuda().eval()


for height in heights:

    kp_load_path = os.path.join(load_path, "brisk", str(height))
    csv_path = "csv_dir/brisk_matches_%s" % str(height) + ".csv"

    query_path = os.path.join(kp_load_path, "query")
    gallery_path = os.path.join(kp_load_path, "gallery")
    query_list = glob.glob(os.path.join(query_path,"*"))
    gallery_list = glob.glob(os.path.join(gallery_path,"*"))

    query_txt_list = sorted(query_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
    gallery_txt_list = sorted(gallery_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))

    query_img_list = get_datasets_list(height, "query_satellite")
    gallery_img_list = get_datasets_list(height, "gallery_drone")

    print(query_txt_list)
    print(query_img_list)

    match_dict = {}
    # print(len(query_txt_list))
    for num in range(len(query_txt_list)):
        # print(num)
        since = time.time()

        query_number = "{:0>4d}".format(num + 1)
        query_txt = glob.glob(os.path.join(query_txt_list[num], "*"))[0]
        query_img = glob.glob(os.path.join(query_img_list[num], "*"))[0]
        print(query_img)
        print(query_txt)
        # break
        kp1 = load_keypoint(query_txt)
        query_img = cv2.imread(query_img)
        des1 = tfeat_utils.describe_opencv(sosnet32, query_img, kp1, patch_size=32, mag_factor=3)
        match_dict[query_number] = []
        # break
        for j in gallery_txt_list:
            print(j)
            gallery_txt_list = glob.glob(os.path.join(j, "*"))
            gallery_txt_list = sorted(gallery_txt_list, key=lambda x: int(re.findall("[0-9]+", x[-6:])[0]))



            for txt in gallery_txt_list:
                kp2, des2 = load_keypoint(txt)
                match_numbers = bf_match(kp1, des1, kp2, des2, cv2.NORM_L2)
                match_dict[query_number].append(match_numbers)
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        df = pd.DataFrame(match_dict)
        df.to_csv(csv_path)
        break