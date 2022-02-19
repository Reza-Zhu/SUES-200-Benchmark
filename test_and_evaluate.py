# -*- coding: utf-8 -*-
import glob
import os
import time
import torch
import scipy.io
import shutil
import numpy as np
import pandas as pd
from torch import nn
from utils import fliplr, load_network, which_view, get_id, get_yaml_value
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
    # print("Testing Start >>>>>>>>")
    table_path = os.path.join(get_yaml_value("weight_save_path"),
                              get_yaml_value("name") + ".csv")
    save_model_list = glob.glob(os.path.join("/media/data1/save_model_weight",
                                             get_yaml_value('name'), "*.pth"))
    if os.path.exists(os.path.join("/media/data1/save_model_weight",
                                   get_yaml_value('name'))) and len(save_model_list) >= 5:
        if not os.path.exists(table_path):
            evaluate_csv = pd.DataFrame(index=["recall@1", "recall@5", "recall@10", "recall@1p", "AP", "time"])
        else:
            evaluate_csv = pd.read_csv(table_path)
            evaluate_csv.index = evaluate_csv["index"]
        for query in ['drone', 'satellite']:
            for seq in range(-5, 0):
                model, net_name = load_network(seq=seq)
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

                height = get_yaml_value("height")
                data_path = get_yaml_value("dataset_path")
                data_path = data_path + "/Testing/{}".format(height)
                image_datasets, data_loader = Create_Testing_Datasets(test_data_path=data_path)

                gallery_path = image_datasets[gallery_name].imgs
                query_path = image_datasets[query_name].imgs

                gallery_label, gallery_path = get_id(gallery_path)
                query_label, query_path = get_id(query_path)

                with torch.no_grad():
                    since = time.time()
                    query_feature = extract_feature(model, data_loader[query_name], which_query)
                    gallery_feature = extract_feature(model, data_loader[gallery_name], which_gallery)
                    time_elapsed = time.time() - since
                    # print('Testing complete in {:.0f}m {:.0f}s'.format(
                    #     time_elapsed // 60, time_elapsed % 60))

                    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label,
                              'gallery_path': gallery_path,
                              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}

                    scipy.io.savemat('pytorch_result.mat', result)

                # print(">>>>>>>> Testing END")
                #
                # print("Evaluating Start >>>>>>>>")

                # if get_yaml_value("query") == "satellite":
                #     query_name = 'satellite'
                #     gallery_name = 'drone'
                # elif get_yaml_value("query") == "drone":
                #     query_name = 'drone'
                #     gallery_name = 'satellite'

                result = scipy.io.loadmat("pytorch_result.mat")

                # initialize query feature data
                query_feature = torch.FloatTensor(result['query_f'])
                query_label = result['query_label'][0]

                # initialize all(gallery) feature data
                gallery_feature = torch.FloatTensor(result['gallery_f'])
                gallery_label = result['gallery_label'][0]

                # fed tensor to GPU
                query_feature = query_feature.cuda()
                gallery_feature = gallery_feature.cuda()

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

                evaluate_csv[query_name+"_"+net_name] = [float(recall_1), float(recall_5),
                                                         float(recall_10), float(recall_1p),
                                                         float(AP), float(time_elapsed)]
                evaluate_result = 'Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f Time:%.2f' % (
                    recall_1, recall_5, recall_10, recall_1p, AP, time_elapsed)

                # show result and save
                save_path = os.path.join(get_yaml_value("weight_save_path"), get_yaml_value('name'))
                save_txt_path = os.path.join(save_path,
                                             '%s_to_%s_%s_%.2f_%.2f.txt' % (query_name[6:], gallery_name[8:], net_name[:7],
                                                                            recall_1, AP))
                # print(save_txt_path)

                with open(save_txt_path, 'w') as f:
                    f.write(evaluate_result)
                    f.close()

                shutil.copy('settings.yaml', os.path.join(save_path, "settings_saved.yaml"))
                # print(round(len(gallery_label)*0.01))
                print(evaluate_result)
        # evaluate_csv["max"] =
        drone_max = []
        satellite_max = []

        for index in evaluate_csv.index:
            drone_max.append(evaluate_csv.loc[index].iloc[:5].max())
            satellite_max.append(evaluate_csv.loc[index].iloc[5:].max())

        evaluate_csv['drone_max'] = drone_max
        evaluate_csv['satellite_max'] = satellite_max
        evaluate_csv.columns.name = "net"
        evaluate_csv.index.name = "index"
        evaluate_csv.to_csv(table_path)
    else:
        print("Don't have enough weights to evaluate!")

if __name__ == '__main__':
    eval_and_test()