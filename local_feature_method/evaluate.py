import os
import pandas as pd
import numpy as np
from compute import generate_index, get_query_list

query = get_query_list()
csv_file = 'query_drone_300.csv'
gallery = pd.read_csv('csv_dir' + os.sep + csv_file)

# query = pd.read_csv('query_list.csv')
# query = query['query'].tolist()

index_list = []
DATASETS_LENGTH = 149
IMG_LIST_LENGTH = 50
ap = 0.0
CMC = np.zeros(len(gallery.index), dtype=np.float32)

# rebuild Dataframe index and column
for i in range(1, DATASETS_LENGTH + 1):
    for j in range(IMG_LIST_LENGTH):
        num = "{:0>4d}".format(i)
        index_list.append(num)

# print(gallery)
gallery['index'] = index_list
gallery = gallery.set_index('index')
gallery.columns = query

# compute recall and mAP
for i in query:
    ap_tmp, CMC_tmp = generate_index(i, gallery)
    if CMC_tmp[0] == -1:
        continue
    CMC += CMC_tmp
    ap += ap_tmp

CMC = CMC / len(query)
result = 'Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
    CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(DATASETS_LENGTH * 0.01)] * 100,
    ap / len(query) * 100)
save_txt_path = 'result_dir' + os.sep + csv_file[:-3] + 'txt'
with open(save_txt_path, 'w') as f:
    f.write(result)
    f.close()
print(result)
