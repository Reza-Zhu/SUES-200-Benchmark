import os
import torch
import numpy as np
from torchvision import models,datasets,transforms
from netvlad import NetVLAD, EmbedNet


def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths


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
    # print("cmc.shape",cmc.shape)
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


encoder = models.resnet18(pretrained=True)
base_model = torch.nn.Sequential(
    encoder.conv1,
    encoder.bn1,
    encoder.relu,
    encoder.maxpool,
    encoder.layer1,
    encoder.layer2,
    encoder.layer3,
    encoder.layer4,
)

dim = list(base_model.parameters())[-1].shape[0]
netVLAD = NetVLAD(num_clusters=89, dim=dim, alpha=1.0)
model = EmbedNet(base_model, netVLAD).cuda()

trans_train_list = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
state_dict = torch.load('./save_model_weight/model_49.pt')
model.load_state_dict(state_dict)

height = "150"
classes = 89
data_path = "/media/data1/Datasets"
test_data_path = data_path + "/Testing/{}".format(height)
batch_size = 16


query_test_datasets = datasets.ImageFolder(os.path.join(test_data_path, "query_drone"),
                                           transform=trans_train_list)

gallery_test_datasets = datasets.ImageFolder(os.path.join(test_data_path, "gallery_satellite"),
                                             transform=trans_train_list)

query_data_loader = torch.utils.data.DataLoader(query_test_datasets,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=8)
gallery_data_loader = torch.utils.data.DataLoader(gallery_test_datasets,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=8)

query_path = query_test_datasets.imgs
gallery_path = gallery_test_datasets.imgs

query_label, query_path = get_id(query_path)
gallery_label, gallery_path = get_id(gallery_path)

query_label_all = np.array(query_label)
gallery_label_all = np.array(gallery_label)

print(query_label_all)
print(gallery_label_all)

query_features = torch.FloatTensor()
gallery_features = torch.FloatTensor()
# gallery_label = get_id(os.path.join(test_data_path, "gallery_satellite"))
with torch.no_grad():
    count = 0
    for gallery_image, gallery_label in gallery_data_loader:
        n, c, h, w = gallery_image.size()
        count += n
        output = model(gallery_image.cuda())
        # print(output_test.shape)
        gallery_features = torch.cat((gallery_features, output.data.cpu()))

    print(gallery_features.size())
    print(count)

    count = 0
    for query_image, query_label in query_data_loader:
        n, c, h, w = query_image.size()
        count += n
        output = model(query_image.cuda())
        query_features = torch.cat((query_features, output.data.cpu()))

    print(query_features.size())
    print(count)

query_feature = query_features.cuda()
gallery_feature = gallery_features.cuda()

# print(query_feature[0].size())
print(query_label_all)
# print(gallery_feature.size())
print(gallery_label_all)
# print()

CMC = torch.IntTensor(len(gallery_label_all)).zero_()
# ap = average precision
ap = 0.0

for i in range(len(query_label_all)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label_all[i], gallery_feature, gallery_label_all)
    # print(CMC_tmp.shape)
    if CMC_tmp[0] == -1:
        continue
    CMC += CMC_tmp
    ap += ap_tmp

CMC = CMC.float()
CMC = CMC / len(query_label_all)

# show result and save
# save_path = os.path.join('save_model_weight', get_yaml_value('name'))
# save_txt_path = os.path.join(save_path, '%s_to_%s_result.txt' % (query_name, gallery_name))
result = 'Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
    CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
    ap / len(query_label_all) * 100)

print(result)
