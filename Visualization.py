import os
import re
import glob
import torch
import model_
import pandas as pd
import numpy as np
from torch import nn
from evaluation_methods import select_best_weight
from utils import get_yaml_value, which_view, get_id, get_best_weight
from test_and_evaluate import extract_feature
from Preprocessing import Create_Testing_Datasets


def get_rank(height, query_name, gallery_name, model_name):
    data_path = get_yaml_value("dataset_path")
    data_path = data_path + "/Testing/{}".format(height)
    gallery_drone_path = os.path.join(data_path, "gallery_drone")
    gallery_satellite_path = os.path.join(data_path, "gallery_satellite")
    gallery_drone_list = glob.glob(os.path.join(gallery_drone_path, "*"))
    gallery_drone_list = sorted(gallery_drone_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
    print(gallery_drone_list)
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

    net_path = get_best_weight(query_name, model_name, height)

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
    print(image_datasets[query_name].imgs)
    for i in range(80):
        query = query_feature[i].view(-1, 1)
        score = torch.mm(gallery_feature, query)
        score = score.squeeze(1).cpu()
        index = np.argsort(score.numpy())
        index = index[::-1].tolist()
        max_score_list = index[0:10]

        for index in max_score_list:
            most_correlative_img = drone_list[index]
            print(most_correlative_img)
        break

    # def paint_heat_map():


if __name__ == '__main__':
    query_name = 'query_satellite'
    gallery_name = 'gallery_drone'
    get_rank(300, 'query_satellite', 'gallery_drone', "resnet")
