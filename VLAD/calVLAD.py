import os
import glob
import pickle
import numpy as np
from VLADlib.VLAD import getVLADDescriptors
from VLADlib import Descriptors


def get_VLADBook(data_path, vb_path, save_path):
    with open(vb_path, 'rb') as f:
        visualDictionary = pickle.load(f)
    data_list = glob.glob(os.path.join(data_path, "*"))
    print(data_list)
    idImage_list = []
    descriptors_list = None
    # print(descriptors_list.shape)
    for i in range(len(data_list)):
        if i == 0:
            descriptors_list, idImages = getVLADDescriptors(data_list[i], Descriptors.describeSIFT, visualDictionary)
            # print(descriptors_list.shape)
            idImage_list += idImages
        else:
            arr, idImages = getVLADDescriptors(data_list[i], Descriptors.describeSIFT, visualDictionary)
            descriptors_list = np.concatenate((descriptors_list, arr), axis=0)
            # print(descriptors_list.shape)
            idImage_list += idImages
        print(len(idImage_list))

        # break
    # print(descriptors_list)

    with open(save_path, 'wb') as f:
        pickle.dump([idImage_list, descriptors_list, save_path], f)

    print("The VLAD descriptors are saved in " + save_path)


if __name__ == '__main__':

    for Height in [150, 200, 250, 300]:

        satellite_data_path = '/media/data1/Datasets/Testing/' + str(Height) + '/query_satellite'
        drone_data_path = '/media/data1/Datasets/Testing/' + str(Height) + '/query_drone'

        satellite_save_path = "./Data/query_satellite_%s_VLADDictionary" % str(Height) + ".pickle"
        drone_save_path = "./Data/query_drone_%s_VLADDictionary" % str(Height) + ".pickle"

        satellite_vb_path = "./Data/satellite_%s_visualDictionary" % str(Height) + ".pickle"
        drone_vb_path = "./Data/drone_%s_visualDictionary" % str(Height) + ".pickle"
        # print(satellite_vb_path)

        get_VLADBook(satellite_data_path, satellite_vb_path, satellite_save_path)
        get_VLADBook(drone_data_path, drone_vb_path, drone_save_path)

    for Height in [150, 200, 250, 300]:
        satellite_data_path = '/media/data1/Datasets/Testing/' + str(Height) + '/gallery_satellite'
        drone_data_path = '/media/data1/Datasets/Testing/' + str(Height) + '/gallery_drone'

        satellite_save_path = "./Data/gallery_satellite_%s_VLADDictionary" % str(Height) + ".pickle"
        drone_save_path = "./Data/gallery_drone_%s_VLADDictionary" % str(Height) + ".pickle"

        satellite_vb_path = "./Data/satellite_%s_visualDictionary" % str(Height) + ".pickle"
        drone_vb_path = "./Data/drone_%s_visualDictionary" % str(Height) + ".pickle"
        # print(satellite_vb_path)

        get_VLADBook(satellite_data_path, satellite_vb_path, satellite_save_path)
        get_VLADBook(drone_data_path, drone_vb_path, drone_save_path)
