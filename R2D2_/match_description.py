import glob
import re
import cv2
import torch
import pandas as pd
import os
import time
from extract import extract_keypoints
from tools import common
from nets.patchnet import *
from local_learning_feature_method.utils import load_keypoint, get_datasets_list
from local_learning_feature_method.matcher import bf_match

# load_path = "/media/tp/data/save_dir"
# load_path = "/media/data1/save_dir"

def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    # print("\n>> Creating net = " + checkpoint['net'])
    net = eval(checkpoint['net'])

    nb_of_weights = common.model_size(net)
    # print(f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return net.eval()


iscuda = common.torch_set_gpu(0)

# load the network...
net = load_network("models/r2d2_WASF_N16.pt")
if iscuda: net = net.cuda()

heights = [150, 200]
# tp :250 300
# sues: 150 200
torch.no_grad()



for height in heights:
    # kp_load_path = os.path.join(load_path, "sift", str(height))
    csv_path = "csv_dir/R2D2_matches_%s" % str(height) + ".csv"
    #
    # query_path = os.path.join(kp_load_path, "query")
    # gallery_path = os.path.join(kp_load_path, "gallery")
    # query_list = glob.glob(os.path.join(query_path,"*"))
    # gallery_list = glob.glob(os.path.join(gallery_path,"*"))
    #
    # query_txt_list = sorted(query_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
    # gallery_txt_list = sorted(gallery_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))

    query_img_list = get_datasets_list(height, "query_satellite")
    gallery_img_list = get_datasets_list(height, "gallery_drone")

    match_dict = {}
    # print(len(query_txt_list))
    for num in range(len(query_img_list)):
        print(num)
        query_number = "{:0>4d}".format(num + 1)
        # query_txt = glob.glob(os.path.join(query_txt_list[num], "*"))[0]
        query_img = glob.glob(os.path.join(query_img_list[num], "*"))[0]

        des1 = extract_keypoints(query_img, net)
        match_dict[query_number] = []
        for j in range(len(gallery_img_list)):
            # print(gallery_img_list[j])
            since = time.time()
            # gallery_txt_list_ = glob.glob(os.path.join(gallery_txt_list[j], "*"))
            gallery_img_list_ = glob.glob(os.path.join(gallery_img_list[j],"*"))
            # gallery_txt_list_ = sorted(gallery_txt_list_, key=lambda x: int(re.findall("[0-9]+", x[-6:])[0]))
            gallery_img_list_ = sorted(gallery_img_list_, key=lambda x: int(re.findall("[0-9]+", x[-6:])[0]))
            for k in range(len(gallery_img_list_)):
                # kp2 = load_keypoint(gallery_txt_list_[k])
                gallery_img = gallery_img_list_[k]
                # gallery_img = cv2.imread(gallery_img, 0)
                des2 = extract_keypoints(gallery_img, net)

                match_numbers = bf_match(de1=des1, de2=des2, distance=cv2.NORM_L2)
                match_dict[query_number].append(match_numbers)

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        df = pd.DataFrame(match_dict)
        df.to_csv(csv_path)
    #     break
    # break