import glob
import os
import pandas as pd
import numpy as np


def compute_mAP(index, good_index, junk_index=None):
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

result_path_list = glob.glob(os.path.join("./result",".csv"))
for result_path in result_path_list:
    result_path = "./result/satellite2drone_150.csv"
    df = pd.read_csv(result_path,index_col='index')

    # print(df.columns)

    index_list = []
    DATASETS_LENGTH = 149
    IMG_LIST_LENGTH = 50
    LABEL_LENGTH = 60
    ap = 0.0
    CMC = np.zeros(len(df.index), dtype=np.float32)

    # rebuild Dataframe index and column
    for i in range(1, DATASETS_LENGTH + 1):
        for j in range(IMG_LIST_LENGTH):
            num = "{:0>4d}".format(i)
            index_list.append(num)

    df['index'] = index_list
    df = df.set_index('index')
    print(df.head())


    for label in df.columns:
        # print(int(i))
        table = df[label].to_numpy()
        sorted_arr = np.argsort(table)
        sorted_arr = sorted_arr[::-1]
        index = np.array(df.index)
        good_index = np.argwhere(index == label[:4])
        ap_tmp, CMC_tmp = compute_mAP(sorted_arr, good_index)
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp
        # break
        print(CMC)

    CMC = CMC / LABEL_LENGTH
    result = 'Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
        CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(DATASETS_LENGTH * 0.01)] * 100,
        ap / LABEL_LENGTH * 100)

    print(result)



