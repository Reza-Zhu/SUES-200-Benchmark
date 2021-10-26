# -*- coding: utf-8 -*-
import os
import torch
import scipy.io
from torch import nn
from utils import fliplr, load_network, which_view, get_id
from Preprocessing import Create_Testing_Datasets


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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
print("Testing Start >>>>>>>>")

model = load_network()
model.classifier.classifier = nn.Sequential()

# 网络模型还需改进
# print(model)

model = model.eval()
model = model.cuda()

query_name = 'query_satellite'
# query_name = 'query_drone'

# gallery_name = 'gallery_satellite'
gallery_name = 'gallery_drone'

which_gallery = which_view(gallery_name)
which_query = which_view(query_name)

print('%d -> %d:' % (which_query, which_gallery))

image_datasets, data_loader = Create_Testing_Datasets()
# print(image_datasets["query_drone"].imgs)

gallery_path = image_datasets[gallery_name].imgs
query_path = image_datasets[query_name].imgs


gallery_label, gallery_path = get_id(gallery_path)
query_label, query_path = get_id(query_path)

with torch.no_grad():
    query_feature = extract_feature(model, data_loader[query_name], which_query)
    gallery_feature = extract_feature(model, data_loader[gallery_name], which_gallery)

    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_path': gallery_path,
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}

    scipy.io.savemat('pytorch_result.mat', result)
    # print(result)
print(">>>>>>>> Testing END")
# os.system('python evaluate.py')
