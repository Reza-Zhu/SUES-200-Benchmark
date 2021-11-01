import glob
import os
import re

import numpy as np
import pandas as pd

def evaluate(number,table):
    table_numpy = table[number].to_numpy()
    all_index = np.argsort(table_numpy)
    all_index = all_index[::-1]
    index = np.array(table.index)
    good_index = np.argwhere(index == number)
    CMC_tmp = compute_mAP(all_index, good_index)
    return CMC_tmp

def compute_mAP(index, good_index, junk_index = None):
    # CMC就是recall的，只要前K里面有一个正确答案就算recall成功是1否则是0
    # mAP是传统retrieval的指标，算的是 recall和precision曲线，这个曲线和x轴的面积。
    # 你可以自己搜索一下mAP

    ap = 0
    cmc = np.zeros(len(index))
    # print(cmc.shape) torch.Size([51355])
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index

    # mask = np.in1d(index, junk_index, invert=True)
    # index = index[mask]

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

df = pd.read_csv('query_drone_star_150.csv')
query = pd.read_csv('query_list.csv')
query = query['query'].tolist()
print(query)
# print(df)
index_list = []
DATASETS_LENGTH = 150
IMG_LIST_LENGTH = 50
for i in range(1, DATASETS_LENGTH):
    for j in range(IMG_LIST_LENGTH):
        num = "{:0>4d}".format(i) \
              # + '-' + "{:0>3d}".format(j)
        # print(num)
        index_list.append(num)
for i in range(len(query)):
    query[i] = "{:0>4d}".format(query[i])
    # print(query[i])
df['index'] = index_list
df = df.set_index('index')
df.columns = query
# df.to_csv("query_indexed.csv")
# print(df)
ap = 0.0
CMC = np.zeros(len(df.index),dtype=np.float32)

# CMC_temp = evaluate('0001',df)
for i in query:
    ap_tmp, CMC_tmp = evaluate(i, df)
    if CMC_tmp[0] == -1:
        continue
    CMC += CMC_tmp
    ap += ap_tmp
print(len(CMC))

CMC = CMC / len(query)


result = 'Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
    CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(149 * 0.01)] * 100,
    ap / len(query) * 100)

print(result)
# print(CMC, ap)
