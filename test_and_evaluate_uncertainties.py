# -*- coding: utf-8 -*-
import glob
import os
import time
import torch
import scipy.io
import shutil
import argparse
import numpy as np
import pandas as pd
from torch import nn
from utils import fliplr, which_view, get_id, get_yaml_value, get_best_weight, parameter, create_dir
from Preprocessing import Create_Testing_Datasets_uncertainties
import model_

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


############################### main function #######################################
def eval_and_test(cfg_path, types, heights, csv_dir_path):
    print("Testing Start >>>>>>>>")

    params = get_yaml_value(cfg_path)
    # weight_save_path = params["weight_save_path"]
    data_path = params["dataset_path"]
    # size = params['image_size']
    # batch_size = params['batch_size']

    for type in types:
        print("current type:", type)
        csv_save_path = os.path.join(csv_dir_path, params['model'])
        create_dir(csv_save_path)
        table_path = os.path.join(csv_save_path, params["model"] + "_" + type + ".csv")

        # Drecall@1 = Drone -> Satellite recall@1
        # DAP       = Drone -> Satellite AP
        # Srecall@1 = Satellite -> Drone recall@1
        # SAP       = Satellite -> Drone AP

        evaluate_csv = pd.DataFrame(index=["Drecall@1_" + type, "DAP_" + type, "Srecall@1_" + type, "SAP_" + type])

        for height in heights:
            parameter('height', height)
            query_drone = []
            query_satellite = []
            data_path_test = data_path + "/Testing/{}".format(height)

            for query in ['drone', 'satellite']:

                # find the best weight of "model"
                net_path = get_best_weight(query, params["model"], height, params["weight_save_path"])
                model = model_.model_dict[params["model"]](params["classes"], params["drop_rate"], pretrained=False)

                print(height, net_path)
                print(params["model"])
                model.load_state_dict(torch.load(net_path))
                model.classifier.classifier = nn.Sequential()

                model = model.eval()
                model = model.cuda()

                query_name = ""
                gallery_name = ""

                if query == "satellite":
                    query_name = 'query_satellite'
                    gallery_name = 'gallery_drone'
                elif query == "drone":
                    query_name = 'query_drone'
                    gallery_name = 'gallery_satellite'

                which_query = which_view(query_name)
                which_gallery = which_view(gallery_name)

                print('%s -> %s:' % (query_name, gallery_name))
                image_datasets, data_loader = Create_Testing_Datasets_uncertainties(test_data_path=data_path_test,
                                                                                    batch_size=params["batch_size"],
                                                                                    image_size=params["image_size"],
                                                                                    gap=50, type=type)

                gallery_path = image_datasets[gallery_name].imgs
                query_path = image_datasets[query_name].imgs

                gallery_label, gallery_path = get_id(gallery_path)
                query_label, query_path = get_id(query_path)

                with torch.no_grad():
                    since = time.time()
                    query_feature = extract_feature(model, data_loader[query_name], which_query)
                    gallery_feature = extract_feature(model, data_loader[gallery_name], which_gallery)
                    time_elapsed = time.time() - since

                # fed tensor to GPU
                query_feature = query_feature.cuda()
                gallery_feature = gallery_feature.cuda()
                query_label = np.array(query_label)
                gallery_label = np.array(gallery_label)

                # CMC = recall
                CMC = torch.IntTensor(len(gallery_label)).zero_()

                # ap = average precision
                ap = 0.0

                for i in range(len(query_label)):
                    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
                    if CMC_tmp[0] == -1:
                        continue
                    CMC += CMC_tmp
                    ap += ap_tmp

                # average CMC
                CMC = CMC.float()
                CMC = CMC / len(query_label)
                # print(len(query_label))
                recall_1 = CMC[0] * 100
                recall_5 = CMC[4] * 100
                recall_10 = CMC[9] * 100
                recall_1p = CMC[round(len(gallery_label) * 0.01)] * 100
                AP = ap / len(query_label) * 100

                evaluate_result = 'Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f Time:%.2f' % (
                    recall_1, recall_5, recall_10, recall_1p, AP, time_elapsed)

                # print(evaluate_result)

                if query == "drone":
                    query_drone = [float(recall_1), AP]
                else:
                    query_satellite = [float(recall_1), AP]
                print(evaluate_result)

            evaluate_csv[height] = [*query_drone, *query_satellite]
        evaluate_csv.to_csv(table_path)



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='settings.yaml', help='config file XXX.yaml path')
    parser.add_argument('--types', type=list, default=["rain"], help='list, choosing uncertainties, '
                                                               '["rain", "fog", "snow", "flip", "black"]')

    parser.add_argument('--heights', type=list, default=[150], help='list, choosing heights, [150, 200, 250, 300]')
    parser.add_argument('--csv_save_path', type=str, default="./result",
                        help="evaluation result table store path")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt(True)
    eval_and_test(opt.cfg, opt.types, opt.heights, opt.csv_save_path)

