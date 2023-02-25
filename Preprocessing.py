import torch
import os
from torchvision import datasets, transforms
from utils import get_yaml_value
from Uncertainties_Imgaug import AddBlock, Weather


def Create_Training_Datasets(train_data_path, batch_size, image_size):
    training_data_loader = {}
    transform_drone_list = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transforms_satellite_list = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    drone_train_datasets = datasets.ImageFolder(os.path.join(train_data_path, "drone"),
                                                transform=transforms.Compose(transform_drone_list))
    satellite_train_datasets = datasets.ImageFolder(os.path.join(train_data_path, "satellite"),
                                                    transform=transforms.Compose(transforms_satellite_list))

    training_data_loader["drone_train"] = torch.utils.data.DataLoader(drone_train_datasets,
                                                                      batch_size=batch_size,
                                                                      shuffle=True,
                                                                      num_workers=4,  # 多进程
                                                                      pin_memory=True)  # 锁页内存

    training_data_loader["satellite_train"] = torch.utils.data.DataLoader(satellite_train_datasets,
                                                                          batch_size=batch_size,
                                                                          shuffle=True,
                                                                          num_workers=4,  # 多进程
                                                                          pin_memory=True)  # 锁页内存

    return training_data_loader

def Create_Testing_Datasets(test_data_path, batch_size, image_size):
    testing_data_loader = {}
    image_datasets = {}

    transforms_test_list = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        # iaa.Sequential([seq]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transforms_list = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    image_datasets['query_drone'] = datasets.ImageFolder(os.path.join(test_data_path, "query_drone"),
                                                         transform=transforms.Compose(transforms_test_list))

    image_datasets['query_satellite'] = datasets.ImageFolder(os.path.join(test_data_path, "query_satellite"),
                                                             transform=transforms.Compose(transforms_test_list))

    image_datasets['gallery_drone'] = datasets.ImageFolder(os.path.join(test_data_path, "gallery_drone"),
                                                           transform=transforms.Compose(transforms_list))

    image_datasets['gallery_satellite'] = datasets.ImageFolder(os.path.join(test_data_path, "gallery_satellite"),
                                                               transform=transforms.Compose(transforms_list))

    testing_data_loader["query_drone"] = torch.utils.data.DataLoader(image_datasets['query_drone'],
                                                                     batch_size=batch_size,
                                                                     shuffle=False,
                                                                     num_workers=4,  # 多进程
                                                                     pin_memory=True)

    testing_data_loader["query_satellite"] = torch.utils.data.DataLoader(image_datasets['query_satellite'],
                                                                         batch_size=batch_size,
                                                                         shuffle=False,
                                                                         num_workers=4,  # 多进程
                                                                         pin_memory=True)  # 锁页内存

    testing_data_loader["gallery_drone"] = torch.utils.data.DataLoader(image_datasets['gallery_drone'],
                                                                       batch_size=batch_size,
                                                                       shuffle=False,
                                                                       num_workers=4,  # 多进程
                                                                       pin_memory=True)  # 锁页内存

    testing_data_loader["gallery_satellite"] = torch.utils.data.DataLoader(image_datasets['gallery_satellite'],
                                                                           batch_size=batch_size,
                                                                           shuffle=False,
                                                                           num_workers=4,  # 多进程
                                                                           pin_memory=True)  # 锁页内存

    return image_datasets, testing_data_loader



def Create_Testing_Datasets_uncertainties(test_data_path, batch_size, image_size, gap, type):
    testing_data_loader = {}
    image_datasets = {}
    if type in ['snow', 'rain', 'fog']:

        transforms_test_list = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            Weather(type),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    elif type in ['flip', 'black']:

        transforms_test_list = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            AddBlock(gap=gap, type=type),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    elif type == "normal" or "":
        transforms_test_list = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    transforms_list = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    image_datasets['query_drone'] = datasets.ImageFolder(os.path.join(test_data_path, "query_drone"),
                                                            transform=transforms.Compose(transforms_test_list),
                                                            )

    image_datasets['query_satellite'] = datasets.ImageFolder(os.path.join(test_data_path, "query_satellite"),
                                                            transform=transforms.Compose(transforms_list),
                                                            )

    image_datasets['gallery_drone'] = datasets.ImageFolder(os.path.join(test_data_path, "gallery_drone"),
                                                           transform=transforms.Compose(transforms_test_list))

    image_datasets['gallery_satellite'] = datasets.ImageFolder(os.path.join(test_data_path, "gallery_satellite"),
                                                               transform=transforms.Compose(transforms_list))

    testing_data_loader["query_drone"] = torch.utils.data.DataLoader(image_datasets['query_drone'],
                                                                     batch_size=batch_size,
                                                                     shuffle=False,
                                                                     num_workers=4,  # 多进程
                                                                     pin_memory=True)

    testing_data_loader["query_satellite"] = torch.utils.data.DataLoader(image_datasets['query_satellite'],
                                                                         batch_size=batch_size,
                                                                         shuffle=False,
                                                                         num_workers=4,  # 多进程
                                                                         pin_memory=True)  # 锁页内存

    testing_data_loader["gallery_drone"] = torch.utils.data.DataLoader(image_datasets['gallery_drone'],
                                                                       batch_size=batch_size,
                                                                       shuffle=False,
                                                                       num_workers=4,  # 多进程
                                                                       pin_memory=True)  # 锁页内存

    testing_data_loader["gallery_satellite"] = torch.utils.data.DataLoader(image_datasets['gallery_satellite'],
                                                                           batch_size=batch_size,
                                                                           shuffle=False,
                                                                           num_workers=4,  # 多进程
                                                                           pin_memory=True)  # 锁页内存

    return image_datasets, testing_data_loader


if __name__ == "__main__":
    params = get_yaml_value("settings.yaml")
    data_path = params["dataset_path"]
    data_path = data_path + "/Testing/{}".format(params["height"])
    image_datasets, data_loader = Create_Testing_Datasets_uncertainties(test_data_path=data_path,
                                                                        batch_size=params["batch_size"],
                                                                        image_size=params["image_size"],
                                                                        gap=50, type="fog")

    for data in data_loader['query_drone']:
        img, label = data
        print(img.shape, label.shape)
        # break