from __future__ import print_function, division

import argparse
import math
from pathlib import Path
import time
import torch
import model_
import shutil
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from Preprocessing import Create_Training_Datasets
from utils import get_device, get_yaml_value, save_network, create_dir


def _next_or_restart(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_one_epoch(model, loaders, criterion, optimizer, scaler, device, use_amp):
    """Run one balanced epoch over the two independent view loaders."""
    model.train()
    satellite_loader = loaders["satellite_train"]
    drone_loader = loaders["drone_train"]
    satellite_iter = iter(satellite_loader)
    drone_iter = iter(drone_loader)
    steps = max(len(satellite_loader), len(drone_loader))
    if steps == 0:
        raise RuntimeError("Training dataloaders produced no batches")

    loss_total = torch.zeros((), device=device)
    satellite_correct = torch.zeros((), dtype=torch.long, device=device)
    drone_correct = torch.zeros((), dtype=torch.long, device=device)
    satellite_total = 0
    drone_total = 0

    for _ in range(steps):
        data1, satellite_iter = _next_or_restart(satellite_loader, satellite_iter)
        data2, drone_iter = _next_or_restart(drone_loader, drone_iter)
        input1, label1 = (value.to(device, non_blocking=True) for value in data1)
        input2, label2 = (value.to(device, non_blocking=True) for value in data2)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            output1, output2 = model(input1, input2)
            loss = criterion(output1, label1) + criterion(output2, label2)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_total += loss.detach()
        satellite_correct += torch.argmax(output1.detach(), dim=1).eq(label1).sum()
        drone_correct += torch.argmax(output2.detach(), dim=1).eq(label2).sum()
        satellite_total += label1.size(0)
        drone_total += label2.size(0)

    return {
        "loss": loss_total.item() / steps,
        "satellite_acc": satellite_correct.item() / satellite_total,
        "drone_acc": drone_correct.item() / drone_total,
    }


def train(config_path, epochs=None):
    param_dict = get_yaml_value(config_path)
    print(param_dict)
    classes = param_dict["classes"]
    num_epochs = epochs if epochs is not None else param_dict["num_epochs"]
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")
    drop_rate = param_dict["drop_rate"]
    lr = param_dict["lr"]
    weight_decay = param_dict["weight_decay"]
    model_name = param_dict["model"]
    fp16 = param_dict["fp16"]
    Batch_size = param_dict["batch_size"]
    size = param_dict["image_size"]
    weight_save_path = param_dict["weight_save_path"]
    device = get_device(param_dict.get("device"))
    cudnn.benchmark = device.type == "cuda"

    train_data_path = Path(param_dict["dataset_path"]) / "Training" / str(param_dict["height"])
    data_loader = Create_Training_Datasets(train_data_path=train_data_path, batch_size=Batch_size,
                                           image_size=size,
                                           num_workers=param_dict.get("num_workers", 4),
                                           pin_memory=param_dict.get("pin_memory", device.type == "cuda"),
                                           prefetch_factor=param_dict.get("prefetch_factor", 2))
    print("Dataloader Preprocessing Finished...")

    model = model_.model_dict[model_name](
        classes,
        drop_rate,
        share_weight=False,
        pretrained=param_dict.get("pretrained", True),
    )
    model = model.to(device)
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * lr},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=weight_decay, momentum=0.9, nesterov=True)

    use_amp = bool(fp16) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        param_dict.get("lr_milestones", [20, 40]),
        gamma=param_dict.get("lr_gamma", 0.1),
    )

    best_loss = float("inf")
    print("Training Start >>>>>>>>")
    weight_save_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    dir_model_name = model_name + "_" + str(param_dict["height"]) + "_" + weight_save_name
    save_path = Path(weight_save_path) / dir_model_name
    create_dir(save_path)
    shutil.copy(config_path, save_path / "settings_saved.yaml")

    for epoch in range(num_epochs):
        since = time.time()
        stats = train_one_epoch(model, data_loader, criterion, optimizer, scaler, device, use_amp)

        scheduler.step()
        epoch_loss = stats["loss"]
        satellite_acc = stats["satellite_acc"]
        drone_acc = stats["drone_acc"]
        time_elapsed = time.time() - since

        print('[Epoch {}/{}] {} | Loss: {:.4f} | Drone_Acc: {:.2f}% | Satellite_Acc: {:.2f}% | Time: {:.2f}s' \
              .format(epoch + 1, num_epochs, "Train", epoch_loss, drone_acc * 100, satellite_acc * 100, time_elapsed))

        if math.isfinite(epoch_loss) and epoch_loss < best_loss:
            best_loss = epoch_loss
            save_network(model, dir_model_name, epoch + 1, weight_save_path=weight_save_path)
            print(model_name + " Epoch: " + str(epoch + 1) + " has saved with loss: " + str(epoch_loss))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='settings.yaml', help='config file XXX.yaml path')
    parser.add_argument('--epochs', type=int, default=None, help='override num_epochs for a controlled run')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt(True)
    print(opt.cfg)
    train(opt.cfg, opt.epochs)
