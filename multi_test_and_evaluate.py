# -*- coding: utf-8 -*-
import glob
import os
import time
import model_
import torch
import scipy.io
import shutil
import numpy as np
import pandas as pd
from torch import nn
from utils import fliplr, load_network, which_view, get_id, get_yaml_value, get_best_weight
from Preprocessing import Create_Testing_Datasets

if torch.cuda.is_available():
    device = torch.device("cuda:0")

def evaluate(qf, ql, gf, gl):
    # print(qf.shape) torch.Size([512])
    # print(gf.shape) torch.Size([51355, 512])
    # print(ql) 0 ()
    # print(gl) [0,0...0] len = 51355 shape = (51355,)

    query = qf.view(-1, 1)
    # print(query.shape)  query.shape = (512,1)
    # gf.shape = (51355, 512)
    # 矩阵相乘

    # score 是否可理解为当前余弦距离的排序？
    score = torch.mm(gf, query)
    # score.shape = (51355,1)
    score = score.squeeze(1).cpu()
    # score.shape = （51355,)
    score = score.numpy()
    # print(score)
    # print(score.shape)

    # predict index
    index = np.argsort(score)  # from small to large
    # 从小到大的索引排列
    # print("index before", index)
    index = index[::-1]
    # print("index after", index)
    # 从大到小的索引排列

    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    # print(query_index.shape) (54, 1)
    # gl = ql 返回标签值相同的索引矩阵
    # 得到 ql：卫星图标签，gl：无人机图标签
    # 即 卫星图标签在 gl中的索引位置 组成的矩阵
    good_index = query_index

    # print(good_index)
    # print(index[0:10])
    junk_index = np.argwhere(gl == -1)
    # print(junk_index)  = []

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    # CMC就是recall的，只要前K里面有一个正确答案就算recall成功是1否则是0
    # mAP是传统retrieval的指标，算的是 recall和precision曲线，这个曲线和x轴的面积。
    # 你可以自己搜索一下mAP

    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    # print(cmc.shape) torch.Size([51355])
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    # print(index.shape) (51355,)
    # if junk_index == []
    # return index fully

    # find good_index index
    ngood = len(good_index)
    # print("good_index", good_index) (54, 1)
    # print(index)
    # print(good_index)
    mask = np.in1d(index, good_index)
    # print(mask)
    # print(mask.shape)  (51355,)
    # 51355 中 54 个对应元素变为了True

    rows_good = np.argwhere(mask == True)
    # print(rows_good.shape) (54, 1)
    # rows_good 得到这 54 个为 True 元素的索引位置

    rows_good = rows_good.flatten()
    # print(rows_good.shape)  (54,)
    # print(rows_good[0])

    cmc[rows_good[0]:] = 1
    # print(cmc)
    # print(cmc.shape) torch.Size([51355])

    # print(cmc)
    for i in range(ngood):
        d_recall = 1.0 / ngood
        # d_racall = 1/54
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        # n/sum
        # print("row_good[]", i, rows_good[i])
        # print(precision)
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
            # ”2范数“ 也称为Euclid范数（欧几里得范数，常用计算向量长度），
            # 即：向量元素绝对值的平方和再开方，表示x到零点的欧式距离

            ff = ff.div(fnorm.expand_as(ff))
            # 把fnorm扩展成ff一样的形状，提高维度，
            # div除法（逐元素相除）

        features = torch.cat((features, ff.data.cpu()), 0)  # 在维度0上拼接
    return features


############################### main function #######################################
def eval_and_test():
    # heights = [150, 200, 250, 300]
    #
    # multi = True
    # model_name = "seresnet"
    # query_name = "query_drone"
    # gallery_name = "gallery_satellite"
    #
    # data_path = "/media/data1/Datasets"
    # for height in heights:
    #
    #     data_path = data_path + "/Testing/{}".format(height)
    #     net_path = get_best_weight(query_name, model_name, height)
    #     model = model_.model_dict[model_name](120, 0)
    #     model.load_state_dict(torch.load(net_path))
    #     model.classifier.classifier = nn.Sequential()
    #     model = model.eval()
    #     model = model.cuda()
    #     which_query = which_view(query_name)
    #     which_gallery = which_view(gallery_name)
    #     image_datasets, data_loader = Create_Testing_Datasets(test_data_path=data_path)
    #     gallery_path = image_datasets[gallery_name].imgs
    #     query_path = image_datasets[query_name].imgs
    #
    #     gallery_label, gallery_path = get_id(gallery_path)
    #     query_label, query_path = get_id(query_path)
    #
    #     with torch.no_grad():
    #
    #         query_feature = extract_feature(model, data_loader[query_name], which_query)
    #
    #         gallery_feature = extract_feature(model, data_loader[gallery_name], which_gallery)
    #
    #         result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label,
    #                   'gallery_path': gallery_path, 'query_f': query_feature.numpy(),
    #                   'query_label': query_label, 'query_path': query_path}
    #
    #         scipy.io.savemat('multi_pytorch_result.mat', result)
    #         print("multi_pytorch_result.mat has saved")

        result = scipy.io.loadmat("multi_pytorch_result.mat")

        # initialize query feature data
        query_feature = torch.FloatTensor(result['query_f'])
        query_label = result['query_label'][0]

        # initialize all(gallery) feature data
        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_label = result['gallery_label'][0]

        # fed tensor to GPU
        query_feature = query_feature.cuda()
        new_query_feature = torch.FloatTensor().cuda()
        gallery_feature = gallery_feature.cuda()
        multi = True
        coff = 2
        image_per_class = len(query_label) // (200 - get_yaml_value("classes")) // coff
        print(image_per_class)
        query_length = len(query_label) + image_per_class

        feature_list = list(range(0, query_length, image_per_class))
        query_concat = np.ones(((len(feature_list)-1)//2, 2))

        if multi:
            query_label = np.intersect1d(query_label, gallery_label)
            for i in range(len(query_label)):
                query_concat[i] = query_label[i] * query_concat[i]

            query_label = query_concat.reshape(-1,)
            for i in range(len(feature_list)):
                if feature_list[i] == (query_length - image_per_class):
                    continue
                multi_feature = torch.mean(query_feature[feature_list[i]:feature_list[i+1], :], 0).view(1, 512)
                new_query_feature = torch.cat((new_query_feature, multi_feature), 0)

            query_feature = new_query_feature

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


        CMC = CMC.float()
        CMC = CMC / len(query_label)
        # print(len(query_label))
        recall_1 = CMC[0] * 100
        recall_5 = CMC[4] * 100
        recall_10 = CMC[9] * 100
        recall_1p = CMC[round(len(gallery_label) * 0.01)] * 100
        AP = ap / len(query_label) * 100

        evaluate_result = 'Recall@1:%.4f Recall@5:%.4f Recall@10:%.4f Recall@top1:%.4f AP:%.4f' % (
            recall_1, recall_5, recall_10, recall_1p, AP)

        # show result and save
        # save_path = os.path.join(get_yaml_value("weight_save_path"), get_yaml_value('name'))

        # shutil.copy('settings.yaml', os.path.join(save_path, "settings_saved.yaml"))
        # print(round(len(gallery_label)*0.01))
        print(evaluate_result)



if __name__ == '__main__':
    eval_and_test()
# Recall@1:55.00 Recall@5:80.00 Recall@10:82.50 Recall@top1:73.75 AP:59.97
# Recall@1:65.0000 Recall@5:88.7500 Recall@10:92.5000 Recall@top1:80.0000 AP:69.4017
# Recall@1:72.5000 Recall@5:90.0000 Recall@10:93.7500 Recall@top1:82.5000 AP:75.9981
# Recall@1:78.7500 Recall@5:96.2500 Recall@10:100.0000 Recall@top1:91.2500 AP:82.2309