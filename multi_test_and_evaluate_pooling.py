# -*- coding: utf-8 -*-
import glob
import os
import time
import model_
import torch
import scipy.io
import shutil
from einops import rearrange
import argparse
import numpy as np
import pandas as pd
from torch import nn
from utils import fliplr, load_network, which_view, get_id, get_yaml_value, get_best_weight, create_dir, parameter
from Preprocessing import Create_Testing_Datasets

if torch.cuda.is_available():
    device = torch.device("cuda:0")

def evaluate(qf, ql, gf, gl):

    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    junk_index = np.argwhere(gl == -1)
    # print(junk_index)  = []

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):

    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    # print(cmc.shape) torch.Size([51355])
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)

    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1

    for i in range(ngood):
        d_recall = 1.0 / ngood
        # d_racall = 1/54
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def extract_feature(model, dataloaders, view_index=1):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, 512).zero_().cuda()

        # why for in range(2)：
        # 1. for flip img
        # 2. for normal img

        for i in range(2):
            if i == 1:
                img = fliplr(img)

            input_img = img.to(device)
            outputs = None
            if view_index == 1:
                outputs, _ = model(input_img, None)
            elif view_index == 2:
                _, outputs = model(None, input_img)

            ff += outputs
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)  # 在维度0上拼接
    return features


def eval_and_test(multi_coff, config_file, type, save_path):
    param = get_yaml_value(config_file)

    create_dir(save_path)
    table_path = os.path.join(save_path, param["model"] + "_" + str(param['height']) + "_" + "multi_query_" + type +
                              ".csv")
    evaluate_csv = pd.DataFrame(index=["recall@1", "recall@5", "recall@10", "recall@1p", "AP", "time"])

    # coff = 1  query images = 50/1 = 50
    # coff = 2  query images = 50/2 = 25

    query_name = "query_drone"
    gallery_name = "gallery_satellite"

    save_model_path = param["weight_save_path"]
    data_path = param["dataset_path"]
    data_path = data_path + "/Testing/{}".format(param["height"])

    net_path = get_best_weight(query_name, param["model"], param["height"], save_model_path)
    model = model_.model_dict[param["model"]](120, 0)
    model.load_state_dict(torch.load(net_path))

    model.classifier.classifier = nn.Sequential()
    model = model.eval()
    model = model.cuda()
    which_query = which_view(query_name)
    which_gallery = which_view(gallery_name)
    image_datasets, data_loader = Create_Testing_Datasets(data_path, param['batch_size'],
                                                          param["image_size"])
    gallery_path = image_datasets[gallery_name].imgs
    query_path = image_datasets[query_name].imgs

    gallery_label, gallery_path = get_id(gallery_path)
    query_label, query_path = get_id(query_path)

    with torch.no_grad():
        since = time.time()
        query_feature = extract_feature(model, data_loader[query_name], which_query)
        gallery_feature = extract_feature(model, data_loader[gallery_name], which_gallery)
        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    query_label = np.array(query_label)
    gallery_label = np.array(gallery_label)

    # fed tensor to GPU
    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    image_per_class = len(query_label) // (200 - param["classes"]) // multi_coff
    query_length = len(query_label) + image_per_class

    feature_list = list(range(0, query_length, image_per_class))
    query_concat = np.ones(((len(feature_list)-1)//multi_coff, multi_coff))

    query_label = np.intersect1d(query_label, gallery_label)
    for i in range(len(query_label)):

        query_concat[i] = query_label[i] * query_concat[i]

    query_label = query_concat.reshape(-1,)

    # pooling
    query_feature = rearrange(query_feature, "h w -> w h")

    if type == "max":
        # Max pooling
        m = torch.nn.MaxPool1d(image_per_class)
    elif type == "ave":
        # Average pooling
        m = torch.nn.AvgPool1d(image_per_class)

    query_feature = m(query_feature)
    query_feature = rearrange(query_feature, "h w -> w h")
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)
    # print(len(query_label))
    recall_1 = CMC[0] * 100
    recall_5 = CMC[4] * 100
    recall_10 = CMC[9] * 100
    recall_1p = CMC[round(len(gallery_label) * 0.01)] * 100
    AP = ap / len(query_label) * 100
    evaluate_csv["multi_query" + "_" + str(image_per_class) +
                 "_" + str(param["height"])] = \
        [float(recall_1), float(recall_5),
         float(recall_10), float(recall_1p),
         float(AP), float(0)]

    print(evaluate_csv)

    evaluate_csv.columns.name = "height"
    evaluate_csv.index.name = "index"
    evaluate_csv = evaluate_csv.T
    evaluate_csv.to_csv(table_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='settings.yaml', help='config file XXX.yaml path')
    parser.add_argument('--multi', type=int, default=1, help='multi number for example: if multi == 1 fusion image '
                                                             'number = 50/1 = 50')
    parser.add_argument('--type', type=str, default="ave", help='feature ensemble strategy, '
                                                                'ave: average pooling or max: max pooling')
    parser.add_argument('--csv_save_path', type=str, default="./result", help="evaluation result table store path")
    opt = parser.parse_known_args()[0]

    eval_and_test(opt.multi, opt.cfg, opt.type, opt.csv_save_path)
