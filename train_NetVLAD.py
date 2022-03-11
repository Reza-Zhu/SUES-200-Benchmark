import os
import time
import torch
import torch.nn as nn
from torchvision import models
from utils import get_yaml_value, save_network
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

num_epochs = get_yaml_value("num_epochs")
height = get_yaml_value("height")
classes = get_yaml_value("classes")
model_name = "NetVLAD"

lr = 0.01
dim = list(base_model.parameters())[-1].shape[0]
netVLAD = NetVLAD(num_clusters=classes, dim=dim, alpha=1.0)
model = EmbedNet(base_model, netVLAD).cuda()

criterion = HardTripletLoss(margin=0.1).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9, nesterov=True)
# scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

data_loader = Create_Training_Datasets()
print("<<<<<<<<<Training Start>>>>>>>>>>>>")
current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

min_loss = 0.005
for epoch in range(num_epochs):
    running_loss = 0.0
    total = 0.0
    model.train(True)
    # for data1, data2 in zip(data_loader["satellite_train"], data_loader["drone_train"]):
    for batch_dix, (input1, label1) in enumerate(data_loader["drone_train"]):

        # input1, label1 = data1
        # input2, label2 = data2
        # print("label1:", label1)
        # print("label2:", label2)

        input1 = input1.cuda()
        label1 = label1.cuda()
        total += label1.size(0)
        optimizer.zero_grad()
        output1 = model(input1)
        print(output1.shape)
        loss = criterion(output1, label1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # print('[ In Epoch {}-{}] {} | Loss: {:.8f} | '\
        #       .format(epoch + 1, batch_dix, "Train", loss.item()))
    # scheduler.step()
    epoch_loss = running_loss/total
    print('<<<<[Epoch {}/{}] {} | Loss: {:.8f} |>>>>'\
          .format(epoch + 1, num_epochs, "Train", epoch_loss))
    save_dir = "/media/data1/save_model_weight/" + model_name+"_"+current_time
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if epoch_loss < min_loss:
        min_loss = epoch_loss
        save_network(model, model_name+"_"+current_time, epoch + 1)
        print(model_name + "Epoch: " + str(epoch + 1) + " has saved with loss: " + str(epoch_loss))

        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # torch.save(model.state_dict(),
        #            os.path.join(save_dir, "net_%03d.pth" % epoch))
