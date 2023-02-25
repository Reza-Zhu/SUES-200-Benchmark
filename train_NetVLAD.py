import os
import time
import torch
import torch.nn as nn
from torchvision import models
from utils import get_yaml_value, save_network, parameter, create_dir
from torch.optim import lr_scheduler
from NetVLAD.netvlad import NetVLAD, EmbedNet
from NetVLAD.tripleloss import HardTripletLoss
from Preprocessing import Create_Training_Datasets

encoder = models.resnet18(pretrained=True)
base_model = nn.Sequential(
    encoder.conv1,
    encoder.bn1,
    encoder.relu,
    encoder.maxpool,
    encoder.layer1,
    encoder.layer2,
    encoder.layer3,
    encoder.layer4,
)

param_dict = get_yaml_value("settings.yaml")

num_epochs = param_dict["num_epochs"]
height = param_dict["height"]
classes = param_dict["classes"]
Batch_size = param_dict["batch_size"]
size = param_dict["image_size"]
model_name = "NetVLAD"

lr = 0.01
dim = list(base_model.parameters())[-1].shape[0]
netVLAD = NetVLAD(num_clusters=classes, dim=dim, alpha=1.0)
model = EmbedNet(base_model, netVLAD).cuda()

criterion = HardTripletLoss(margin=0.1).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9, nesterov=True)
# scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

train_data_path = param_dict["dataset_path"] + "/Training/{}".format(param_dict["height"])
data_loader = Create_Training_Datasets(train_data_path=train_data_path, batch_size=Batch_size,
                                       image_size=size)
print("<<<<<<<<<Training Start>>>>>>>>>>>>")
current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
save_dir = model_name + "_" + current_time
parameter("name", save_dir)
dir_path = os.path.join(param_dict["weight_save_path"], save_dir)
create_dir(dir_path)

min_loss = 0.005
for epoch in range(num_epochs):
    running_loss = 0.0
    total = 0.0
    model.train(True)
    # for data1, data2 in zip(data_loader["satellite_train"], data_loader["drone_train"]):
    for batch_dix, (input1, label1) in enumerate(data_loader["drone_train"]):

        input1 = input1.cuda()
        label1 = label1.cuda()
        total += label1.size(0)
        optimizer.zero_grad()
        output1 = model(input1)
        # print(output1.shape)
        loss = criterion(output1, label1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss/total
    print('<<<<[Epoch {}/{}] {} | Loss: {:.8f} |>>>>'\
          .format(epoch + 1, num_epochs, "Train", epoch_loss))


    if epoch_loss < min_loss:
        min_loss = epoch_loss
        save_network(model, save_dir, epoch + 1)
        print(model_name + "Epoch: " + str(epoch + 1) + " has saved with loss: " + str(epoch_loss))

