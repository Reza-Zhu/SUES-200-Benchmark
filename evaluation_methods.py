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
    ap_drone_height = {}
    ap_satellite_height = {}
    for csv in model_csv_list:
        height = csv.split("/")[-1].split("_")[1]
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

        ap_drone_height[height] = drone_table["AP"].max()
        ap_satellite_height[height] = satellite_table["AP"].max()

    delta_drone_AP1 = (ap_drone_height["200"] - ap_drone_height["150"])/50
    delta_drone_AP2 = (ap_drone_height["250"] - ap_drone_height["200"])/50
    delta_drone_AP3 = (ap_drone_height["300"] - ap_drone_height["250"])/50
    delta_drone_list = [delta_drone_AP1, delta_drone_AP2, delta_drone_AP3]
    average_drone_AP = sum(delta_drone_list)/len(delta_drone_list)
    stability_drone = {"delta_drone_AP": delta_drone_list, "average_drone_AP": average_drone_AP}

    delta_satellite_AP1 = (ap_satellite_height["200"] - ap_satellite_height["150"])/50
    delta_satellite_AP2 = (ap_satellite_height["250"] - ap_satellite_height["300"])/50
    delta_satellite_AP3 = (ap_satellite_height["300"] - ap_satellite_height["250"])/50
    delta_satellite_list = [delta_satellite_AP1, delta_satellite_AP2, delta_satellite_AP3]
    average_satellite_AP = sum(delta_satellite_list)/len(delta_drone_list)
    stability_satellite = {"delta_drone_AP": delta_satellite_list, "average_drone_AP": average_satellite_AP}

    return stability_drone, stability_satellite

if __name__ == '__main__':
    print(evaluate_stability("resnet"))
    # adaption = evaluate_adaption_rate("resnet")
    # print(adaption)
