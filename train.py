from __future__ import print_function, division

import argparse
import time
import yaml
import torch
import model_
import shutil
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from Preprocessing import Create_Training_Datasets
from utils import get_yaml_value, save_network, parameter, create_dir
from test_and_evaluate import eval_and_test
from model_ import ClassBlock
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
cudnn.benchmark = True


def train(config_path):
    # config_path = "settings.yaml."
    param_dict = get_yaml_value(config_path)
    print(param_dict)
    classes = param_dict["classes"]
    num_epochs = param_dict["num_epochs"]
    drop_rate = param_dict["drop_rate"]
    lr = param_dict["lr"]
    weight_decay = param_dict["weight_decay"]
    model_name = param_dict["model"]
    fp16 = param_dict["fp16"]
    Batch_size = param_dict["batch_size"]
    size = param_dict["image_size"]
    weight_save_path = param_dict["weight_save_path"]

    train_data_path = param_dict["dataset_path"] + "/Training/{}".format(param_dict["height"])
    data_loader = Create_Training_Datasets(train_data_path=train_data_path, batch_size=Batch_size,
                                           image_size=size)
    print("Dataloader Preprocessing Finished...")

    model = model_.model_dict[model_name](classes, drop_rate, share_weight=False, pretrained=True)

    model.classifier = ClassBlock(768, classes, drop_rate)
    model = model.cuda()
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * lr},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=weight_decay, momentum=0.9, nesterov=True)

    if fp16:
        # from apex.fp16_utils import *
        try:
            from apex import amp, optimizers
            model, optimizer_ft = amp.initialize(model, optimizer, opt_level="O2")

        except ImportError:
            print("please install apex")
            fp16 = 0

    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.1)

    MAX_LOSS = 1
    print("Training Start >>>>>>>>")
    weight_save_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    dir_model_name = model_name + "_" + str(param_dict["height"]) + "_" + weight_save_name
    save_path = os.path.join(weight_save_path, dir_model_name)
    create_dir(save_path)
    parameter("name", dir_model_name)
    shutil.copy(config_path, os.path.join(save_path, "settings_saved.yaml"))

    for epoch in range(num_epochs):
        since = time.time()

        running_loss = 0.0
        running_corrects1 = 0.0
        running_corrects2 = 0.0
        total1 = 0.0
        total2 = 0.0
        model.train(True)
        for data1, data2 in zip(data_loader["satellite_train"], data_loader["drone_train"]):
            input1, label1 = data1
            input2, label2 = data2

            input1 = input1.to(device, non_blocking=True)
            input2 = input2.to(device, non_blocking=True)
            label1 = label1.to(device, non_blocking=True)
            label2 = label2.to(device, non_blocking=True)

            optimizer.zero_grad()

            output1, output2 = model(input1, input2)
            _, preds1 = torch.max(output1.data, 1)
            _, preds2 = torch.max(output2.data, 1)
            total1 += label1.size(0)
            total2 += label2.size(0)
            loss1 = criterion(output1, label1)
            loss2 = criterion(output2, label2)

            loss = loss1 + loss2

            if fp16:  # we use optimizer to backward loss
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects1 += preds1.eq(label1.data).sum()
            running_corrects2 += preds2.eq(label2.data).sum()

        scheduler.step()
        epoch_loss = running_loss / classes
        satellite_acc = running_corrects1 / total1
        drone_acc = running_corrects2 / total2
        time_elapsed = time.time() - since

        print('[Epoch {}/{}] {} | Loss: {:.4f} | Drone_Acc: {:.2f}% | Satellite_Acc: {:.2f}% | Time: {:.2f}s' \
              .format(epoch + 1, num_epochs, "Train", epoch_loss, drone_acc * 100, satellite_acc * 100, time_elapsed))

        if drone_acc > 0.97 and satellite_acc > 0.97:
            if epoch_loss < MAX_LOSS and epoch > (num_epochs - 20):
                MAX_LOSS = epoch_loss
                save_network(model, dir_model_name, epoch + 1)
                print(model_name + " Epoch: " + str(epoch + 1) + " has saved with loss: " + str(epoch_loss))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='settings.yaml', help='config file XXX.yaml path')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt(True)
    print(opt.cfg)
    train(opt.cfg)

