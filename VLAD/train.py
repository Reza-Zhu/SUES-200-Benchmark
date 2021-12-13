import os
import glob
import pickle
import numpy as np

from VLADlib.VLAD import getDescriptors, kMeansDictionary
from VLADlib import Descriptors


def get_visualBook(data_path, K, save_path):
    data_list = glob.glob(os.path.join(data_path, "*"))
    descriptors_list = None
    # print(descriptors_list.shape)
    for i in range(len(data_list)):
        if i == 0:
            descriptors_list = getDescriptors(data_list[i], Descriptors.describeSIFT)
        else:
            arr = getDescriptors(data_list[i], Descriptors.describeSIFT)
            descriptors_list = np.concatenate((descriptors_list, arr), axis=0)
            print(descriptors_list.shape)
    print(descriptors_list.shape)
    np.save("descriptors", descriptors_list)
    visualDictionary = kMeansDictionary(descriptors_list, K)
    print(visualDictionary)
    # file_path = "./Data/satellite_visualDictionary"+".pickle"

    with open(save_path, 'wb') as f:
        pickle.dump(visualDictionary, f)


for Height in [300]:
    # Height = 150
    K = 128
    satellite_data_path = '/media/data1/Datasets/Training/' + str(Height) + '/satellite'
    drone_data_path = '/media/data1/Datasets/Training/' + str(Height) + '/drone'
    satellite_save_path = "./Data/satellite_%s_visualDictionary" % str(Height) + ".pickle"
    drone_save_path = "./Data/drone_%s_visualDictionary" % str(Height) + ".pickle"

    # if Height == 250:
    get_visualBook(drone_data_path, K, drone_save_path)
    # else:
    #     get_visualBook(drone_data_path, K, drone_save_path)
    #     get_visualBook(satellite_data_path, K, satellite_save_path)
