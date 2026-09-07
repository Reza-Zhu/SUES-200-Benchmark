import os

import torch
from torchvision import datasets, transforms


NORMALIZE = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _make_loader(dataset, batch_size, shuffle, num_workers, pin_memory, prefetch_factor=2):
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset contains no readable images: {dataset.root}")
    num_workers = max(0, int(num_workers))
    batch_size = max(1, int(batch_size))
    loader_options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True
        if prefetch_factor:
            loader_options["prefetch_factor"] = int(prefetch_factor)
    return torch.utils.data.DataLoader(dataset, **loader_options)


def _resize_normalize(image_size):
    return [
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZE),
    ]


def Create_Training_Datasets(
    train_data_path, batch_size, image_size, num_workers=4, pin_memory=True, prefetch_factor=2
):
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZE),
        ]
    )
    drone_dataset = datasets.ImageFolder(
        os.path.join(train_data_path, "drone"), transform=train_transform
    )
    satellite_dataset = datasets.ImageFolder(
        os.path.join(train_data_path, "satellite"), transform=train_transform
    )
    return {
        "drone_train": _make_loader(
            drone_dataset, batch_size, True, num_workers, pin_memory, prefetch_factor
        ),
        "satellite_train": _make_loader(
            satellite_dataset, batch_size, True, num_workers, pin_memory, prefetch_factor
        ),
    }


def Create_Testing_Datasets(
    test_data_path, batch_size, image_size, num_workers=4, pin_memory=True, prefetch_factor=2
):
    test_transform = transforms.Compose(_resize_normalize(image_size))
    image_datasets = {
        name: datasets.ImageFolder(
            os.path.join(test_data_path, name), transform=test_transform
        )
        for name in ("query_drone", "query_satellite", "gallery_drone", "gallery_satellite")
    }
    testing_data_loader = {
        name: _make_loader(dataset, batch_size, False, num_workers, pin_memory, prefetch_factor)
        for name, dataset in image_datasets.items()
    }
    return image_datasets, testing_data_loader


def Create_Testing_Datasets_uncertainties(
    test_data_path,
    batch_size,
    image_size,
    gap,
    type,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
):
    from Uncertainties_Imgaug import AddBlock, Weather

    if type in ("snow", "rain", "fog"):
        query_transforms = _resize_normalize(image_size)[:1] + [
            Weather(type),
            *_resize_normalize(image_size)[1:],
        ]
    elif type in ("flip", "black"):
        query_transforms = _resize_normalize(image_size)[:1] + [
            AddBlock(gap=gap, type=type),
            *_resize_normalize(image_size)[1:],
        ]
    elif type in ("normal", ""):
        query_transforms = _resize_normalize(image_size)
    else:
        raise ValueError(f"Unsupported uncertainty type: {type}")

    normal_transform = transforms.Compose(_resize_normalize(image_size))
    query_transform = transforms.Compose(query_transforms)
    image_datasets = {
        "query_drone": datasets.ImageFolder(
            os.path.join(test_data_path, "query_drone"), transform=query_transform
        ),
        "query_satellite": datasets.ImageFolder(
            os.path.join(test_data_path, "query_satellite"), transform=normal_transform
        ),
        "gallery_drone": datasets.ImageFolder(
            os.path.join(test_data_path, "gallery_drone"), transform=query_transform
        ),
        "gallery_satellite": datasets.ImageFolder(
            os.path.join(test_data_path, "gallery_satellite"), transform=normal_transform
        ),
    }
    testing_data_loader = {
        name: _make_loader(dataset, batch_size, False, num_workers, pin_memory, prefetch_factor)
        for name, dataset in image_datasets.items()
    }
    return image_datasets, testing_data_loader


if __name__ == "__main__":
    from utils import get_yaml_value

    params = get_yaml_value("settings.yaml")
    data_path = os.path.join(params["dataset_path"], "Testing", str(params["height"]))
    _, data_loader = Create_Testing_Datasets_uncertainties(
        test_data_path=data_path,
        batch_size=params["batch_size"],
        image_size=params["image_size"],
        gap=50,
        type="fog",
    )
    for images, labels in data_loader["query_drone"]:
        print(images.shape, labels.shape)
        break
