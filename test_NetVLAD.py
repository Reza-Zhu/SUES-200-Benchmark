import os
import torch
import shutil
import numpy as np
from utils import get_yaml_value, get_id, get_model_list
from evaluate import evaluate
from Preprocessing import Create_Testing_Datasets
from torchvision import models
from NetVLAD.netvlad import NetVLAD, EmbedNet


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

name = get_yaml_value("name")
save_dirname = os.path.join("./save_model_weight", name)
last_model_name = os.path.basename(get_model_list(save_dirname, 'net', -1))
print(last_model_name)
model.load_state_dict(torch.load(os.path.join(save_dirname, last_model_name)))

query_name = 'query_drone'
gallery_name = 'gallery_satellite'

image_datasets, data_loader = Create_Testing_Datasets()
query_path = image_datasets[query_name].imgs
gallery_path = image_datasets[gallery_name].imgs

gallery_label, gallery_path = get_id(gallery_path)
query_label, query_path = get_id(query_path)

query_label = np.array(query_label)
gallery_label = np.array(gallery_label)

query_feature = torch.FloatTensor()
gallery_feature = torch.FloatTensor()

print("<<<<<<<<<Testing Start>>>>>>>>>>>>")

with torch.no_grad():
    for img, label in data_loader[gallery_name]:
        n, c, h, w = img.size()
        output = model(img.cuda())
        gallery_feature = torch.cat((gallery_feature, output.data.cpu()))
    print(gallery_feature.size())
    for img, label in data_loader[query_name]:
        n, c, h, w = img.size()
        output = model(img.cuda())
        query_feature = torch.cat((query_feature, output.data.cpu()))
    print(query_feature.size())


print("<<<<<<<<<Evaluating Start>>>>>>>>>>>>")
CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = average precision
ap = 0.0

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

for i in range(len(query_label)):
    # print(query_label[])
    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
    # print(CMC_tmp.shape)
    if CMC_tmp[0] == -1:
        continue
    CMC += CMC_tmp
    ap += ap_tmp

CMC = CMC.float()
CMC = CMC / len(query_label)
result = 'Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top10:%.2f AP:%.2f' % (
    CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.1)] * 100,
    ap / len(query_label) * 100)

save_path = os.path.join('save_model_weight', get_yaml_value('name'))
save_txt_path = os.path.join(save_path,
                             '%s_to_%s_%s_%.2f_%.2f.txt' % (query_name[6:], gallery_name[8:], last_model_name[:7],
                                                            CMC[0] * 100, ap / len(query_label)*100))

with open(save_txt_path, "w") as f:
    f.write(result)
    f.close()

shutil.copy("settings.yaml", os.path.join(save_path, "settings_saved.yaml"))
print(result)
