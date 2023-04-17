import os
import glob
import shutil
import random
import yaml
import argparse


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_datasets(path, origin_data_path):
    num = path[-4:]
    src_path = os.path.join(origin_data_path, num)
    shutil.copytree(src_path, path)



parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/sues/media/disk1/RAW_DATASETS/SUES-200', help='dataset path')

opt = parser.parse_known_args()[0]

# 原始数据地址
raw_datasets_path = opt.path
video_name = ["150", "200", "250", "300"]

f = open("../indexs.yaml", 'r', encoding="utf-8")
t_value = yaml.load(f, Loader=yaml.FullLoader)
datasets = ["{:0>4d}".format(i+1) for i in range(200)]

train_indexes = t_value["index"]
test_indexes = []
for index in datasets:
    if index not in train_indexes:
        test_indexes.append(index)

Training_path = os.path.join(raw_datasets_path, "Training")
Testing_path = os.path.join(raw_datasets_path, "Testing")

create_dir(Training_path)
create_dir(Testing_path)

for name in glob.glob(os.path.join(raw_datasets_path, "*")):
    if "drone" in name:
        drone_name = os.path.basename(name)

for index_name in video_name:
    satellite_data_path = os.path.join(raw_datasets_path, "satellite-view")
    drone_data_path = os.path.join(raw_datasets_path, drone_name, index_name)

    Training_index_path = os.path.join(Training_path, index_name)
    Testing_index_path = os.path.join(Testing_path, index_name)

    Training_drone_data_list = [os.path.join(Training_index_path, "drone", i) for i in train_indexes]
    Training_satellite_data_list = [os.path.join(Training_index_path, "satellite", i) for i in train_indexes]

    create_dir(Training_index_path)
    create_dir(Testing_index_path)

    # Training dataset
    create_dir(os.path.join(Training_index_path, "drone"))
    create_dir(os.path.join(Training_index_path, "satellite"))

    print("Copying... " + index_name + "m training set of satellite")
    for satellite_index_path in Training_satellite_data_list:
        create_datasets(satellite_index_path, satellite_data_path)

    print("Copying... " + index_name + "m training set of drone")
    for drone_index_path in Training_drone_data_list:
        create_datasets(drone_index_path, drone_data_path)

    # Testing dataset
    Testing_drone_data_list = [os.path.join(Testing_index_path, "query_drone", i) for i in test_indexes]
    Testing_satellite_data_list = [os.path.join(Testing_index_path, "query_satellite", i) for i in test_indexes]

    print("Copying... " + index_name + "m testing set of query satellite")
    for satellite_index_path in Testing_satellite_data_list:
        create_datasets(satellite_index_path, satellite_data_path)

    print("Copying... " + index_name + "m testing set of query drone")

    for drone_index_path in Testing_drone_data_list:
        create_datasets(drone_index_path, drone_data_path)

    Testing_drone_data_list = [os.path.join(Testing_index_path, "gallery_drone", i) for i in datasets]
    Testing_satellite_data_list = [os.path.join(Testing_index_path, "gallery_satellite", i) for i in datasets]

    print("Copying... " + index_name + "m testing set of gallery satellite")

    for satellite_indaex_path in Testing_satellite_data_list:
        create_datasets(satellite_index_path, satellite_data_path)

    print("Copying... " + index_name + "m testing set of gallery drone ")

    for drone_index_path in Testing_drone_data_list:
        create_datasets(drone_index_path, drone_data_path)
