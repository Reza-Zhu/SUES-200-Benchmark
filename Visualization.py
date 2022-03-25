import os
import re
import glob
import torch
import model_
import random
import pandas as pd
import numpy as np
from torch import nn
from evaluation_methods import select_best_weight
from utils import get_yaml_value, which_view, get_id, get_best_weight
from test_and_evaluate import extract_feature
from Preprocessing import Create_Testing_Datasets


def get_rank(height, query_name, gallery_name, model_name, csv_path):
    data_path = get_yaml_value("dataset_path")
    data_path = data_path + "/Testing/{}".format(height)
    gallery_drone_path = os.path.join(data_path, "gallery_drone")
    gallery_satellite_path = os.path.join(data_path, "gallery_satellite")
    gallery_drone_list = glob.glob(os.path.join(gallery_drone_path, "*"))
    gallery_drone_list = sorted(gallery_drone_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
    # print(gallery_drone_list)

    gallery_satellite_list = glob.glob(os.path.join(gallery_satellite_path, "*"))
    gallery_satellite_list = sorted(gallery_satellite_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
    drone_list = []
    satellite_list = []

    if "drone" in gallery_name:
        for drone_img in gallery_drone_list:
            img_list = glob.glob(os.path.join(drone_img, "*"))
            img_list = sorted(img_list, key=lambda x: int(re.findall("[0-9]+", x.split('/')[-1])[0]))
            for img in img_list:
                drone_list.append(img)
    elif "satellite" in gallery_name:
        for satellite_img in gallery_satellite_list:
            img_list = glob.glob(os.path.join(satellite_img, "*"))
            img_list = sorted(img_list, key=lambda x: int(re.findall("[0-9]+", x.split('/')[-1])[0]))
            for img in img_list:
                satellite_list.append(img)

    image_datasets, data_loader = Create_Testing_Datasets(test_data_path=data_path)
    net_path = get_best_weight(query_name, model_name, height, csv_path)
    which_query = which_view(query_name)
    which_gallery = which_view(gallery_name)
    print(net_path)
    model = model_.model_dict[model_name](120, 0)
    model.load_state_dict(torch.load(net_path))
    model.classifier.classifier = nn.Sequential()
    model = model.eval()
    model = model.cuda()
    query_feature = extract_feature(model, data_loader[query_name], which_query)
    gallery_feature = extract_feature(model, data_loader[gallery_name], which_gallery)
    query_img_list = image_datasets[query_name].imgs
    matching_table = {}
    random_sample_list = random.sample(range(0, len(query_img_list)), 10)
    print(random_sample_list)
    for i in random_sample_list:
        query = query_feature[i].view(-1, 1)
        score = torch.mm(gallery_feature, query)
        score = score.squeeze(1).cpu()
        index = np.argsort(score.numpy())
        index = index[::-1].tolist()
        max_score_list = index[0:10]
        query_img = query_img_list[i][0]
        most_correlative_img = []
        for index in max_score_list:
            if "satellite" in query_name:
                most_correlative_img.append(drone_list[index])
            elif "drone" in query_name:
                most_correlative_img.append(satellite_list[index])
        matching_table[query_img] = most_correlative_img
    matching_table = pd.DataFrame(matching_table)
    print(matching_table)
    save_path = "result/" + query_name.split("_")[-1] + "_" + model_name + "_" + str(height) + "_matching.csv"
    matching_table.to_csv(save_path)

def summary_csv_extract_pic(csv_path):
    csv_table = pd.read_csv(csv_path)
    print(csv_table)
    # csv_table[]


if __name__ == '__main__':
    query_name = 'query_satellite'
    gallery_name = 'gallery_drone'
    model_list = ["resnet", "seresnet", "dense"]
    csv_path = "/media/data1/save_model_weight"
    for model in model_list:
        for height in [150, 200, 250, 300]:
            get_rank(height, query_name, gallery_name, model, csv_path)