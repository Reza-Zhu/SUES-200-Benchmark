from __future__ import print_function, division

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from Preprocessing import Create_Training_Datasets
from utils import get_yaml_value,save_network
import model_
import matplotlib.pyplot as plt
import os


if torch.cuda.is_available():
    device = torch.device("cuda:0")
cudnn.benchmark = True

classes = get_yaml_value("classes")
num_epochs = get_yaml_value("num_epochs")
drop_rate = get_yaml_value("drop_rate")
lr = get_yaml_value("lr")

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []

y_err = {}
y_err['train'] = []
y_err['val'] = []

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    # ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    # ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig('train.jpg')

# model = models.resnet18(pretrained=True)
# full_connection = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Linear(full_connection, 128),
#     nn.ReLU(),
#     nn.Dropout(0.7),
#     nn.Linear(256, Classes),
#     # nn.LogSoftmax(dim=1)
# )
model_name = get_yaml_value("model")
model = model_.model_dict[model_name](classes,drop_rate)
model = model.cuda()
# print(model)

# ignored_params = list(map(id, model.classifier.parameters()))
# base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9, esterov=True)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.MultiStepLR(optimizer, [10, 50], 0.08)
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

data_loader = Create_Training_Datasets()
print("Dataloader Preprocessing Finished...")

total = 0
print("Training Start >>>>>>>>")
model_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 10)
    running_loss = 0.0
    running_corrects = 0.0
    running_corrects2 = 0.0
    running_corrects3 = 0.0
    total1 = 0.0
    total2 = 0.0
    model.train(True)
    for data1, data2 in zip(data_loader["satellite_train"],data_loader["drone_train"]):

        input1, label1 = data1
        input2, label2 = data2

        now_batch_size, c, h, w = input1.shape
        input1 = input1.to(device)
        input2 = input2.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)

        optimizer.zero_grad()
        output1, output2 = model(input1, input2)
        _, preds1 = torch.max(output1.data, 1)
        _, preds2 = torch.max(output2.data, 1)
        # print(label1.size(0))
        # print(label2.size(0))
        total1 += label1.size(0)
        total2 += label2.size(0)
        # print(total1)
        # print(total2)
        loss = criterion(output1, label1) + criterion(output2, label2)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += preds1.eq(label1.data).sum()
        running_corrects2 += preds2.eq(label2.data).sum()
        # print(running_loss)
        # print(running_corrects2)
        # print(running_corrects)

    epoch_loss = running_loss / classes
    epoch_acc = running_corrects / total1
    epoch_acc2 = running_corrects2 / total2

    y_loss['train'].append(epoch_loss)
    y_err['train'].append(epoch_acc.cpu())

    print('{} Loss: {:.4f} Drone_Acc: {:.4f} Satellite_Acc: {:.4f}' \
          .format("Train", epoch_loss, epoch_acc, epoch_acc2))
    draw_curve(epoch)
    if (epoch + 1) % 5 == 0:
        save_network(model, model_name, epoch + 1)