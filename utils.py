import os
import sys
import yaml
import torch
import model_


def get_yaml_value(key_name, file_name="settings.yaml"):
    f = open(file_name, 'r', encoding="utf-8")
    t_value = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    params = t_value[key_name]
    return params


def save_network(network, dir_name, epoch_label):
    dict_name = {"name": dir_name}
    with open("settings.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dict_name, f)
        
    if not os.path.isdir('./save_model_weight/' + dir_name):
        os.mkdir('./save_model_weight/'+dir_name)

    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./save_model_weight',dir_name, save_filename)
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


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s' % dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name



def load_network():
    model_name = get_yaml_value("model")
    name = get_yaml_value("name")
    dirname = os.path.join('./save_model_weight', name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    # print(os.path.join(dirname,last_model_name))
    classes = get_yaml_value("classes")
    drop_rate = get_yaml_value("drop_rate")
    model = model_.model_dict[model_name](classes, drop_rate)
    # model = model_.ResNet(classes, drop_rate)
    model.load_state_dict(torch.load(os.path.join(dirname, last_model_name)))
    return model

def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths