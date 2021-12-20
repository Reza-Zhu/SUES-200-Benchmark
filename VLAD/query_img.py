import os
import glob
import pickle
import pandas as pd
import numpy as np
from VLADlib.VLAD import query


def query2table(K, vb_path, tree_path, dataset_path, table_save_path):
    with open(vb_path, "rb") as vb:
        visualDictionary = pickle.load(vb)

    with open(tree_path, "rb") as td:
        tree = pickle.load(td)[1]

    total_df = {}
    dataset_path = glob.glob(os.path.join(dataset_path, "*"))
    for images_num in range(len(dataset_path)):
        print(dataset_path[images_num])
        images = glob.glob(os.path.join(dataset_path[images_num], "*"))
        query_label = dataset_path[images_num][-4:]
        print(query_label)

        for i in range(len(images)):
            df = pd.DataFrame()
            dist, ind = query(images[i], K, "SIFT", visualDictionary, tree)
            dist = dist.reshape(-1)
            ind = ind.reshape(-1)
            query_label_count = "{}-{}".format(query_label, str(i))
            df[query_label_count] = dist
            df["index"] = ind
            df = df.sort_values(by="index", axis=0, ascending=True)
            # df.index = df["index"]
            df = df.drop("index", axis=1)
            # total_df[query_label_count] = df[query_label_count]
            total_df[query_label_count] = df.to_numpy().reshape(-1)
        # print(total_df)
    total_df = pd.DataFrame(total_df)
    total_df.index.name = "index"
    total_df.columns.name = "query"
    print(total_df)

    total_df.to_csv(table_save_path+".csv")
    # break


if __name__ == '__main__':
    for Height in [150, 200, 250, 300]:
        # Height = 150
        query_satellite_path = "./Data/query_satellite_%s_VLADDictionary" % str(Height) + ".pickle"
        query_drone_path = "./Data/query_drone_%s_VLADDictionary" % str(Height) + ".pickle"

        satellite_vb_path = "./Data/satellite_%s_visualDictionary" % str(Height) + ".pickle"
        drone_vb_path = "./Data/drone_%s_visualDictionary" % str(Height) + ".pickle"

        tree_satellite_path = "./Data/tree_satellite_%s" % str(Height) + ".pickle"
        tree_drone_path = "./Data/tree_drone_%s" % str(Height) + ".pickle"

        satellite_data_path = '/media/data1/Datasets/Testing/' + str(Height) + '/query_satellite'
        drone_data_path = '/media/data1/Datasets/Testing/' + str(Height) + '/query_drone'

        query2table(K=149, vb_path=drone_vb_path, tree_path=tree_satellite_path,
                    dataset_path=drone_data_path, table_save_path="./result/drone2satellite_%s" % str(Height))

        query2table(K=7450, vb_path=satellite_vb_path, tree_path=tree_drone_path,
                    dataset_path=satellite_data_path, table_save_path="./result/satellite2drone_%s" % str(Height))
