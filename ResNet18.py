import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import torch.backends.cudnn as cudnn
import numpy as np
# import model
import matplotlib.pyplot as plt
import copy
import time
import os
import math
from shutil import copyfile
from model_ import *
from autoaugment import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")

train_data_path = "../MyData/TrainData"
test_data_path = "../MyData/TestData"
BatchSize = 64*4
Epoch_Num = 50
std = [0.485, 0.456, 0.406]
mean = [0.229, 0.224, 0.225]


def imshow(img):
    img = img.numpy()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    img = img * 255
    img = np.transpose(img, (1, 2, 0))
    # plt.imshow(img)
    plt.imshow(img)
    plt.savefig("img.png")


transform_train_list = [
    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop((384, 384)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    # transforms.RandomRotation((-180, 180)),
    ReIDPolicy(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]

transform_test_list = [
    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'test': transforms.Compose(transform_test_list),
}

train_datasets = datasets.ImageFolder(train_data_path,
                                      transform=data_transforms['train'])
test_datasets = datasets.ImageFolder(test_data_path,
                                     transform=data_transforms['test'])
# print(train_datasets)
len_train = len(train_datasets)
len_test = len(test_datasets)
train_data_loader = DataLoader(train_datasets,
                               batch_size=BatchSize,
                               shuffle=True,
                               num_workers=2,
                               pin_memory=True)

# data_iter = iter(train_data_loader)
# img, lab = data_iter.next()
# imshow(make_grid(img))
# print(lab)

test_data_loader = DataLoader(test_datasets,
                              batch_size=BatchSize,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)

Net = models.resnet18(pretrained=True)
# print(Net)
# Net.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# pthfile_path = "./resnet50-0676ba61.pth"
# Net.load_state_dict(torch.load(pthfile_path))
full_connection = Net.fc.in_features
Net.fc = nn.Sequential(
    nn.Linear(full_connection, 128),
    nn.ReLU(),
    nn.Dropout(0.7),
    nn.Linear(128, 40),
    # nn.LogSoftmax(dim=1)
)
Net = Net.cuda()
# print(Net)
lr = 0.05
optimizer = optim.SGD(Net.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 50], 0.08)
# criterion = nn.CrossEntropyLoss()

# criterion = FocalLoss(class_num=40)
criterion = CELoss(0.05, 40)
warm_up = 0.1  # We start from the 0.1*lrRate
warm_epoch = 5
warm_iteration = round(len(train_datasets)/BatchSize)*warm_epoch

best_epoch = 0
best_acc = 0
train_loss = []
train_acc = []
test_acc = []
for epoch in range(Epoch_Num):
    Net.train(True)
    LOSS = 0
    Train_ACC = 0
    Test_ACC = 0
    for step, (inputs, labels) in enumerate(train_data_loader):
        length = len(train_data_loader)
        # print(inputs,labels)
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = Net(inputs)
        # loss1 = criterion1(outputs, labels)
        loss = criterion(outputs, labels)
        # loss2 = criterion2(outputs, labels)
        warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
        loss *= warm_up

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).sum()
        # acc = torch.mean(correct_counts.type(torch.FloatTensor))

        print('[epoch:%d, iter:%d/%d] Loss: %.03f | Acc: %.3f%% | Best Acc: %.3f%% | Best Epoch: %d'
              % (epoch + 1, (step + 1), int(len_train / BatchSize) + 1,
                 sum_loss / (step + 1), 100. * correct / total,
                 best_acc, best_epoch))
        LOSS += (sum_loss / (step + 1))
        Train_ACC += (100. * correct / total)
    train_loss.append(LOSS)
    train_acc.append(Train_ACC / 207)
    scheduler.step()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        Net.eval()
        for step, (inputs, labels) in enumerate(test_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = Net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum()
        test_acc_ = 100 * test_correct / test_total
        if test_acc_ > best_acc:
            best_epoch = epoch + 1
            best_acc = test_acc_
            now = time.strftime("%H:%M:%S", time.localtime())
            torch.save(Net.state_dict(), './Resnet18_Epoch_%d_Acc_%03d_time:%s.pth' % (best_epoch, best_acc, now))
            sm = torch.jit.script(Net)

            sm.save("./Resnet18_Epoch_%d_Acc_%03d_time:%s.pt" % (best_epoch, best_acc, now))
        print('测试分类准确率为：%.3f%%' % test_acc_)
        test_acc.append(test_acc_)

epoch_list = []
for i in range(Epoch_Num):
    epoch_list.append(i)

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
ax0.plot(epoch_list, train_loss, color='red')
ax1.plot(epoch_list, train_acc, color='red')
ax1.plot(epoch_list, test_acc, color='blue')
plt.savefig("resnet18.png")



