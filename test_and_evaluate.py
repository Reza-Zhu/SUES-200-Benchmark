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
from utils import fliplr, load_network, which_view, get_id, get_yaml_value
from Preprocessing import Create_Testing_Datasets


if torch.cuda.is_available():
    device = torch.device("cuda:0")

def evaluate(qf, ql, gf, gl, dist):

    # Eu Distance
    if "Eu" == dist:
        query = qf.view(1, -1)
        En_dist = nn.PairwiseDistance(p=2)
        score = En_dist(query, gf).cpu()
        index = np.argsort(score)  # from small to large

    elif "Man" == dist:
        # Man Distance
        query = qf.view(1, -1)
        En_dist = nn.PairwiseDistance(p=1)
        score = En_dist(query, gf).cpu()
        index = np.argsort(score)  # from small to large

    # Cosine Distance
    elif "Cos" in dist:
        query = qf.view(-1, 1)
        score = torch.mm(gf, query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
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
def eval_and_test(cfg_path, name, seqs, dist):
    print("Testing Start >>>>>>>>")

    params = get_yaml_value(cfg_path)
    # weight_save_path = params["weight_save_path"]
    data_path = params["dataset_path"]
    data_path = data_path + "/Testing/{}".format(params["height"])
    # size = params['image_size']
    # batch_size = params['batch_size']
    if name == "":
        name = params["name"]
    table_path = os.path.join(params["weight_save_path"], name + ".csv")
    save_model_list = glob.glob(os.path.join(params["weight_save_path"], name, "*.pth"))

    if os.path.exists(os.path.join(params["weight_save_path"], name)) and \
            len(save_model_list) >= 1:

        if not os.path.exists(table_path):
            evaluate_csv = pd.DataFrame(index=["recall@1", "recall@5", "recall@10", "recall@1p", "AP", "time"])
        else:
            evaluate_csv = pd.read_csv(table_path)
            evaluate_csv.index = evaluate_csv["index"]
        for query in ['drone', 'satellite']:
            for seq in range(-seqs, 0):
                model, net_name = load_network(model_name=params["model"], name=name,
                                               weight_save_path=params["weight_save_path"], classes=params["classes"],
                                               drop_rate=params["drop_rate"], seq=seq)

                print(net_name)
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
                image_datasets, data_loader = Create_Testing_Datasets(test_data_path=data_path,
                                                                      batch_size=params["batch_size"],
                                                                      image_size=params["image_size"])

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
                    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label, dist)
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

                evaluate_csv[query_name+"_"+net_name] = [float(recall_1), float(recall_5),
                                                         float(recall_10), float(recall_1p),
                                                         float(AP), float(time_elapsed)]
                evaluate_result = 'Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f Time:%.2f' % (
                    recall_1, recall_5, recall_10, recall_1p, AP, time_elapsed)

                # show result and save
                save_path = os.path.join(params["weight_save_path"], name)
                save_txt_path = os.path.join(save_path,
                                             '%s_to_%s_%s_%.2f_%.2f.txt' % (query_name[6:], gallery_name[8:], net_name[:7],
                                                                            recall_1, AP))

                with open(save_txt_path, 'w') as f:
                    f.write(evaluate_result)
                    f.close()

                print(evaluate_result)
        # evaluate_csv["max"] =
        drone_max = []
        satellite_max = []
        query_number = len(list(filter(lambda x: "drone" in x, evaluate_csv.columns)))

        for index in evaluate_csv.index:

            drone_max.append(evaluate_csv.loc[index].iloc[:query_number].max())
            satellite_max.append(evaluate_csv.loc[index].iloc[query_number:].max())

        evaluate_csv['drone_max'] = drone_max
        evaluate_csv['satellite_max'] = satellite_max
        evaluate_csv.columns.name = "net"
        evaluate_csv.index.name = "index"
        evaluate_csv.to_csv(table_path)
    else:
        print("Don't have enough weights to evaluate!")


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='settings.yaml', help='config file XXX.yaml path')
    parser.add_argument('--name', type=str, default='', help='evaluate which model, name')
    parser.add_argument('--seq', type=int, default=1, help='evaluate how many weights from loss(small -> big)')
    parser.add_argument('--dist', type=str, default='Cos', help='feature distance algorithm: Cosine(Cos), '
                                                                'Euclidean(Eu), Manhattan(Man)')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt(True)

    eval_and_test(opt.cfg, opt.name, opt.seq, opt.dist)
