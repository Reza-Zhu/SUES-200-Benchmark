import os
import glob
import shutil
import random
import argparse

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_datasets(path, data_list, character):
    count = 0
    for i in data_list:
        count = count + 1
        num = i[-4:]
        character_path = os.path.join(path, character)
        # print(character_path)
        path_ = os.path.join(character_path, num)
        # print("src",i)
        # print("dst",path_)
        shutil.copytree(i, path_)
    print(character, " processed ", count, " classes ......")


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../../Datasets/SUES-200', help='dataset path')
parser.add_argument('--coff', type=float, default=0.6, help='Split coff')

opt = parser.parse_known_args()[0]

# 数据集分割系数
SPLIT_VALUE = opt.coff
# 原始数据地址
raw_datasets_path = opt.path
video_name = ["150", "200", "250", "300"]

Training_path = os.path.join(raw_datasets_path, "Training")
Testing_path = os.path.join(raw_datasets_path, "Testing")

create_dir(Training_path)
create_dir(Testing_path)

# origin data
for index_name in video_name:
    satellite_data_path = os.path.join(raw_datasets_path, "satellite-view")
    drone_data_path = os.path.join(raw_datasets_path, "drone-view", index_name)
    print(drone_data_path)

    satellite_data_list = glob.glob(os.path.join(satellite_data_path, "*"))
    print(satellite_data_list)
    drone_data_list = glob.glob(os.path.join(drone_data_path, "*"))
    print(drone_data_list)
    sorted(drone_data_list)
    sorted(satellite_data_list)

    train_data_num = int(len(drone_data_list) * SPLIT_VALUE)
    test_data_num = int(len(drone_data_list) * (1 - SPLIT_VALUE))

    print("training data number: ", train_data_num)
    print("test data number: ", test_data_num)

    Training_satellite_data_list = satellite_data_list[:train_data_num]
    Training_drone_data_list = drone_data_list[:train_data_num]
    Testing_satellite_data_list = satellite_data_list[train_data_num:]
    Testing_drone_data_list = drone_data_list[train_data_num:]

    Training_index_path = os.path.join(Training_path, index_name)
    Testing_index_path = os.path.join(Testing_path, index_name)

    create_dir(Training_index_path)
    create_dir(Testing_index_path)
    #
    ## Training dataset
    create_datasets(Training_index_path, Training_drone_data_list, "drone")
    create_datasets(Training_index_path, Training_satellite_data_list, "satellite")
    #
    ## Testing dataset
    query_drone_path = os.path.join(Testing_index_path, "query_drone")
    query_satellite_path = os.path.join(Testing_index_path, "query_satellite")
    gallery_drone_path = os.path.join(Testing_index_path, "gallery_drone")
    gallery_satellite_path = os.path.join(Testing_index_path, "gallery_satellite")

    create_datasets(query_drone_path, Testing_drone_data_list, "")
    create_datasets(query_satellite_path, Testing_satellite_data_list, "")

    create_datasets(gallery_drone_path, drone_data_list, "")
    create_datasets(gallery_satellite_path, satellite_data_list, "")