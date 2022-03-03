import os
import re
import glob
import torch
import model_
import pandas as pd
import numpy as np
from torch import nn
from evaluation_methods import select_best_weight
from utils import get_yaml_value, which_view, get_id
from test_and_evaluate import extract_feature
from Preprocessing import Create_Testing_Datasets


def get_rank(height, query_name, gallery_name, model_name):
    data_path = get_yaml_value("dataset_path")
    data_path = data_path + "/Testing/{}".format(height)
    gallery_drone_path = os.path.join(data_path, "gallery_drone")
    gallery_satellite_path = os.path.join(data_path, "gallery_satellite")
    gallery_drone_list = glob.glob(os.path.join(gallery_drone_path, "*"))
    gallery_drone_list = sorted(gallery_drone_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
    gallery_satellite_list = glob.glob(os.path.join(gallery_satellite_path, "*"))
    gallery_satellite_list = sorted(gallery_satellite_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
    drone_list = []
    satellite_list = []

    if "drone" in query_name:
        for drone_img in gallery_drone_list:
            img_list = glob.glob(os.path.join(drone_img, "*"))
            img_list = sorted(img_list, key=lambda x: int(re.findall("[0-9]+", x.split('/')[-1])[0]))
            for img in img_list:
                drone_list.append(img)
    elif "satellite" in query_name:
        for satellite_img in gallery_satellite_list:
            img_list = glob.glob(os.path.join(satellite_img, "*"))
            img_list = sorted(img_list, key=lambda x: int(re.findall("[0-9]+", x.split('/')[-1])[0]))
            for img in img_list:
                satellite_list.append(img)

    # print(drone_list)
    # print(satellite_list)
    image_datasets, data_loader = Create_Testing_Datasets(test_data_path=data_path)

    drone_best_list, satellite_best_list = select_best_weight(model_name)
    net_path = None
    if "drone" in query_name:
        for weight in drone_best_list:
            if str(height) in weight:
                drone_best_weight = weight.split(".")[0]
                table = pd.read_csv(weight, index_col=0)
                values = list(table.loc["recall@1", :])[:5]
                indexes = list(table.loc["recall@1", :].index)[:5]
                net_name = indexes[values.index(max(values))]
                net = net_name.split("_")[2] + "_" + net_name.split("_")[3]
                net_path = os.path.join(drone_best_weight, net)
                # print(values, indexes)
    if "satellite" in query_name:
        for weight in satellite_best_list:
            if str(height) in weight:
                satellite_best_weight = weight.split(".")[0]
                table = pd.read_csv(weight, index_col=0)
                values = list(table.loc["recall@1", :])[5:10]
                indexes = list(table.loc["recall@1", :].index)[5:10]
                net_name = indexes[values.index(max(values))]
                net = net_name.split("_")[2] + "_" + net_name.split("_")[3]
                net_path = os.path.join(satellite_best_weight, net)
    # for weight in satellite_best_list:
    #     if str(height) in weight:
    #         satellite_best_weight = weight.split(".")[0]
    #
    # print(drone_best_list)
    # gallery_label, gallery_path = get_id(gallery_path)
    # query_label, query_path = get_id(query_path)

    which_query = which_view(query_name)
    which_gallery = which_view(gallery_name)

    model = model_.model_dict[model_name](120, 0)
    model.load_state_dict(torch.load(net_path))
    model.classifier.classifier = nn.Sequential()
    print(model)
    # model = model.eval()
    # model = model.cuda()
    query_feature = extract_feature(model, data_loader[query_name], which_query)
    gallery_feature = extract_feature(model, data_loader[gallery_name], which_gallery)

    for i in range(80):
        query = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query)
        score = score.squeeze(1).cpu()
        index = np.argsort(score.numpy())
        index = index[::-1]
        print(index)
        # query_index = np.argwhere(gl == ql)
        score = score.numpy().tolist()
        max_score_index = score.index(min(score))
        # index = np.argsort(score)
        print(max_score_index)
        most_correlative_img = drone_list[max_score_index]
        print(most_correlative_img)
        break

    # def paint_heat_map():


if __name__ == '__main__':
    query_name = 'query_satellite'
    gallery_name = 'gallery_drone'
    get_rank(150, 'query_satellite', 'gallery_drone', "resnet")
