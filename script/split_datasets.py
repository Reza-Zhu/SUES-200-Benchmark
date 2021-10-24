import os
import glob
import shutil
import random


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
    print(character," processed ",count," classes ......")


# 数据集分割系数
SPLIT_VALUE = 0.8

# 原始数据地址
raw_datasets_path = "../../RAW_DATASETS"
video_name = ["150", "200", "250", "300"]

Training_path = os.path.join(raw_datasets_path, "Training")
Testing_path = os.path.join(raw_datasets_path, "Testing")

create_dir(Training_path)
create_dir(Testing_path)

# origin data
for index_name in video_name:
    satellite_data_path = os.path.join(raw_datasets_path, "satellite-view")
    drone_data_path = os.path.join(raw_datasets_path, index_name)

    satellite_data_list = glob.glob(os.path.join(satellite_data_path, "*"))
    drone_data_list = glob.glob(os.path.join(drone_data_path, "*"))
    sorted(drone_data_list)
    sorted(satellite_data_list)
    # random.shuffle(satellite_data_list)
    # random.shuffle(drone_data_list)
    # print(satellite_data_list)
    # print(drone_data_list)
    train_data_num = int(len(drone_data_list) * SPLIT_VALUE)
    test_data_num = int(len(drone_data_list) * (1 - SPLIT_VALUE))

    Training_satellite_data_list = satellite_data_list[:train_data_num]
    Training_drone_data_list = drone_data_list[:train_data_num]
    Testing_satellite_data_list = satellite_data_list[train_data_num:]
    Testing_drone_data_list = drone_data_list[train_data_num:]

    # print(len(Testing_label_data_list))
    # print(len(Training_sample_data_list))

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