import os
import sys
import yaml
import torch
import model_
import pandas as pd
from evaluation_methods import select_best_weight

def get_yaml_value(key_name, file_name="settings.yaml"):
    f = open(file_name, 'r', encoding="utf-8")
    t_value = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    params = t_value[key_name]
    return params


def save_network(network, dir_model_name, epoch_label):
    save_path = get_yaml_value('weight_save_path')
    # with open("settings.yaml", "r", encoding="utf-8") as f:
    #     dict = yaml.load(f, Loader=yaml.FullLoader)
    #     dict['name'] = dir_model_name
    #     with open("settings.yaml", "w", encoding="utf-8") as f:
    #         yaml.dump(dict, f)

    # if not os.path.isdir(os.path.join(save_path, dir_model_name)):
    #     os.mkdir(os.path.join(save_path, dir_model_name))

    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(save_path, dir_model_name, save_filename)
    torch.save(network.state_dict(), save_path)


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'drone' in name:
        return 2
    else:
        print('unknown view')
    return -1


def get_model_list(dirname, key, seq):
    if os.path.exists(dirname) is False:
        print('no dir: %s' % dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[seq]
    return last_model_name


def load_network(seq):
    model_name = get_yaml_value("model")
    name = get_yaml_value("name")
    weight_save_path = get_yaml_value("weight_save_path")
    dirname = os.path.join(weight_save_path, name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net', seq))
    print(get_model_list(dirname, 'net', seq) + " " + "seq: " + str(seq))
    # print(os.path.join(dirname,last_model_name))
    classes = get_yaml_value("classes")
    drop_rate = get_yaml_value("drop_rate")
    model = model_.model_dict[model_name](classes, drop_rate)
    # model = model_.ResNet(classes, drop_rate)
    model.load_state_dict(torch.load(os.path.join(dirname, last_model_name)))
    return model, last_model_name


def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_best_weight(query_name, model_name, height):
    drone_best_list, satellite_best_list = select_best_weight(model_name)
    if "drone" in query_name:
        for weight in drone_best_list:
            if str(height) in weight:
                drone_best_weight = weight.split(".")[0]
                table = pd.read_csv(weight, index_col=0)
                values = list(table.loc["recall@1", :])[:5]
                indexes = list(table.loc["recall@1", :].index)[:5]
                net_name = indexes[values.index(max(values))]
                net = net_name.split("_")[2] + "_" + net_name.split("_")[3]
                net_path = os.path.join(drone_best_weight, net)
                # print(values, indexes)
    if "satellite" in query_name:
        for weight in satellite_best_list:
            if str(height) in weight:
                satellite_best_weight = weight.split(".")[0]
                table = pd.read_csv(weight, index_col=0)
                values = list(table.loc["recall@1", :])[5:10]
                indexes = list(table.loc["recall@1", :].index)[5:10]
                net_name = indexes[values.index(max(values))]
                net = net_name.split("_")[2] + "_" + net_name.split("_")[3]
                net_path = os.path.join(satellite_best_weight, net)
    return net_path


def parameter(index_name, index_number):
    with open("settings.yaml", "r", encoding="utf-8") as f:
        setting_dict = yaml.load(f, Loader=yaml.FullLoader)
        setting_dict[index_name] = index_number
        print(setting_dict)
        f.close()
        with open("settings.yaml", "w", encoding="utf-8") as f:
            yaml.dump(setting_dict, f)
            f.close()