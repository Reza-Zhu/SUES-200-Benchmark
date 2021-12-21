import glob
import os
import re
import cv2
import yaml
import pickle

def get_yaml_value(key_name, file_name="../settings.yaml"):
    f = open(file_name, 'r', encoding="utf-8")
    t_value = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    params = t_value[key_name]
    return params

def save_keypoint(kp_tuple, save_path):
    data_dict = {}
    key_points = []
    for kp in kp_tuple:
        temp_kp = [kp.pt, kp.size, kp.angle,
                   kp.response, kp.octave,
                   kp.class_id]
        key_points.append(temp_kp)
    data_dict['keypoint'] = key_points
    # data_dict['descriptor'] = des
    with open(os.path.join(save_path), "wb") as f:
        pickle.dump(data_dict, f, 0)
        f.close()


def load_keypoint(load_path):
    kp_tuple = []
    f = open(os.path.join(load_path), "rb")
    img = pickle.load(f)
    for kp in img['keypoint']:
        KeyPoint = cv2.KeyPoint(x=kp[0][0], y=kp[0][1], size=kp[1], angle=kp[2],
                                response=kp[3], octave=kp[4], class_id=kp[5])
        kp_tuple.append(KeyPoint)
    kp_tuple = tuple(kp_tuple)
    # descriptor = img['descriptor']
    return kp_tuple


def get_datasets_list(height, type):

    datasets_path = get_yaml_value("dataset_path")
    test_path = os.path.join(datasets_path, "Testing", str(height), type)
    test_list = glob.glob(os.path.join(test_path, "*"))
    return sorted(test_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    print(get_yaml_value("dataset_path"))
    # get_datasets_list('query_satellite')