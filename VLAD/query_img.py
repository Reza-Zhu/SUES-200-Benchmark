import os
import glob
import pickle
from VLADlib.VLAD import query


Height = 150
query_satellite_path = "./Data/query_satellite_%s_VLADDictionary" % str(Height) + ".pickle"
query_drone_path = "./Data/query_drone_%s_VLADDictionary" % str(Height) + ".pickle"

satellite_vb_path = "./Data/satellite_%s_visualDictionary" % str(Height) + ".pickle"
drone_vb_path = "./Data/drone_%s_visualDictionary" % str(Height) + ".pickle"

tree_satellite_path = "./Data/tree_satellite_%s" % str(Height) + ".pickle"
tree_drone_path = "./Data/tree_drone_%s" % str(Height) + ".pickle"

satellite_data_path = '/media/data1/Datasets/Testing/' + str(Height) + '/query_satellite'
drone_data_path = '/media/data1/Datasets/Testing/' + str(Height) + '/query_drone'

with open(satellite_vb_path, "rb") as sv:
    sv_visualDictionary = pickle.load(sv)

with open(tree_drone_path, "rb") as td:
    td_tree = pickle.load(td)

tree = td_tree[1]

satellite_data_list = glob.glob(os.path.join(satellite_data_path, "*"))
drone_data_list = glob.glob(os.path.join(drone_data_path, "*"))
print(satellite_data_list)
for images_list in satellite_data_list:
    images = glob.glob(os.path.join(images_list, "*"))
    for img in images:
        dist, ind = query(img, 5, "SIFT", sv_visualDictionary, tree)
        print(dist)
        print(ind)
        break
    break

