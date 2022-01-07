import os
import glob
import pandas as pd


def evaluate_adaption_rate(model_name):
    csv_path = "./save_model_weight"
    model_csv_list = glob.glob(os.path.join(csv_path, model_name+"*.csv"))
    evaluate_drone_height = 0
    evaluate_satellite_height = 0
    for csv in model_csv_list:
        # height = csv.split("/")[-1].split("_")[1]
        table = pd.read_csv(csv)
        table.index = table["index"]
        columns = table.columns
        drone_list = []
        satellite_list = []
        for i in columns:
            if "drone" in i:
                drone_list.append(i)
            if "satellite" in i:
                satellite_list.append(i)

        drone_table = table.loc[:, drone_list]
        satellite_table = table.loc[:, satellite_list]

        drone_table = drone_table.T
        satellite_table = satellite_table.T

        max_drone_height_recall1 = drone_table["recall@1"].max()
        max_satellite_height_recall1 = satellite_table["recall@1"].max()
        evaluate_drone_height += max_drone_height_recall1
        evaluate_satellite_height += max_satellite_height_recall1
    return evaluate_satellite_height/evaluate_drone_height


def evaluate_stability(model_name):
    csv_path = "./save_model_weight"
    model_csv_list = glob.glob(os.path.join(csv_path, model_name + "*.csv"))

if __name__ == '__main__':
    adaption = evaluate_adaption_rate("resnet")
    print(adaption)
