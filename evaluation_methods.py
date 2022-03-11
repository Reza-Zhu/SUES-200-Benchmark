import os
import glob
import numpy as np
import pandas as pd


def select_best_weight(model_name):
    csv_path = "/media/data1/save_model_weight"
    model_csv_list = glob.glob(os.path.join(csv_path, model_name + "*.csv"))

    drone_list = []
    satellite_list = []

    csv_150_list = list(filter(lambda i: "150" in i, model_csv_list))
    csv_200_list = list(filter(lambda i: "200" in i, model_csv_list))
    csv_250_list = list(filter(lambda i: "250" in i, model_csv_list))
    csv_300_list = list(filter(lambda i: "300" in i, model_csv_list))
    csv_lists = [csv_150_list, csv_200_list, csv_250_list, csv_300_list]

    for csv_list in csv_lists:
        drone_recall1_max = 0
        drone_csv_index = None
        satellite_recall1_max = 0
        satellite_csv_index = None
        for csv in csv_list:
            table = pd.read_csv(csv, index_col=0)
            drone_recall1 = table.at["recall@1", "drone_max"]
            satellite_recall1 = table.at["recall@1", "satellite_max"]
            if satellite_recall1 > satellite_recall1_max:
                satellite_recall1_max = satellite_recall1
                satellite_csv_index = csv

            if drone_recall1 > drone_recall1_max:
                drone_recall1_max = drone_recall1
                drone_csv_index = csv

        drone_list.append(drone_csv_index)
        satellite_list.append(satellite_csv_index)

    return drone_list, satellite_list


def evaluate_adaption_rate(model_name):
    drone_list, satellite_list = select_best_weight(model_name)
    evaluate_drone_height = 0
    evaluate_satellite_height = 0

    for csv in drone_list:
        table = pd.read_csv(csv, index_col=0)
        max_drone_height_recall1 = table.at["recall@1", "drone_max"]
        evaluate_drone_height += max_drone_height_recall1
    for csv in satellite_list:
        table = pd.read_csv(csv, index_col=0)
        max_satellite_height_recall1 = table.at["recall@1", "satellite_max"]
        evaluate_satellite_height += max_satellite_height_recall1

    return evaluate_satellite_height/evaluate_drone_height


def forming_precision_table(model_list, save_dir):

    for model_name in model_list:
        drone_list, satellite_list = select_best_weight(model_name)
        drone_total_frame = pd.DataFrame()
        satellite_total_frame = pd.DataFrame()
        for csv in drone_list:
            table = pd.read_csv(csv, index_col=0)
            values = list(table.loc["recall@1", :])[:5]
            indexes = list(table.loc["recall@1", :].index)[:5]
            net_name = indexes[values.index(max(values))]
            recall_ap_list = table.loc[:, net_name].drop(["recall@1p"])
            drone_total_frame = pd.concat([drone_total_frame, recall_ap_list], axis=1)

        drone_total_frame.columns = ["150", "200", "250", "300"]
        drone_total_frame = drone_total_frame.T
        drone_total_frame.to_csv(os.path.join(save_dir, model_name+"_drone.csv"))
        print(drone_total_frame)

        for csv in satellite_list:
            table = pd.read_csv(csv, index_col=0)
            values = list(table.loc["recall@1", :])[5:10]
            indexes = list(table.loc["recall@1", :].index)[5:10]
            net_name = indexes[values.index(max(values))]
            recall_ap_list = table.loc[:, net_name].drop(["recall@1p"])
            satellite_total_frame = pd.concat([satellite_total_frame, recall_ap_list], axis=1)

        satellite_total_frame.columns = ["150", "200", "250", "300"]
        satellite_total_frame = satellite_total_frame.T
        satellite_total_frame.to_csv(os.path.join(save_dir, model_name+"_satellite.csv"))
        print(satellite_total_frame)


def evaluate_stability(model_name, evaluation_value):

    drone_list, satellite_list = select_best_weight(model_name)
    print(drone_list)
    print(satellite_list)
    evaluate_value_drone_height = []
    evaluate_value_satellite_height = []
    for csv in drone_list:
        for height in ["150", "200", "250", "300"]:
            if height in csv:
                table = pd.read_csv(csv, index_col=0)
                evaluate_value_drone_height.append(table.at[evaluation_value, "drone_max"])
    print(evaluate_value_drone_height)
    stability_drone = np.std(evaluate_value_drone_height)

    for csv in satellite_list:
        for height in ["150", "200", "250", "300"]:
            if height in csv:
                table = pd.read_csv(csv, index_col=0)
                evaluate_value_satellite_height.append(table.at[evaluation_value, "satellite_max"])
    print(evaluate_value_satellite_height)
    stability_satellite = np.std(evaluate_value_satellite_height)

    # delta_drone_AP1 = (evaluate_value_drone_height["200"] - evaluate_value_drone_height["150"])/50
    # delta_drone_AP2 = (evaluate_value_drone_height["250"] - evaluate_value_drone_height["200"])/50
    # delta_drone_AP3 = (evaluate_value_drone_height["300"] - evaluate_value_drone_height["250"])/50
    # delta_drone_list = [delta_drone_AP1, delta_drone_AP2, delta_drone_AP3]
    # average_drone = sum(delta_drone_list)/len(delta_drone_list)
    # stability_drone = {"delta_drone": delta_drone_list, "average_drone": average_drone}
    #
    # delta_satellite_AP1 = (evaluate_value_satellite_height["200"] - evaluate_value_satellite_height["150"])/50
    # delta_satellite_AP2 = (evaluate_value_satellite_height["250"] - evaluate_value_satellite_height["200"])/50
    # delta_satellite_AP3 = (evaluate_value_satellite_height["300"] - evaluate_value_satellite_height["250"])/50
    # delta_satellite_list = [delta_satellite_AP1, delta_satellite_AP2, delta_satellite_AP3]
    # average_satellite = sum(delta_satellite_list)/len(delta_satellite_list)
    # stability_satellite = {"delta_drone": delta_satellite_list, "average_drone": average_satellite}

    return stability_drone, stability_satellite


def evaluate_realtime(model_name):
    drone_list, satellite_list = select_best_weight(model_name)
    time_height_dict = {}
    for csv in drone_list:
        height = csv.split("_")[-2]
        time_ave_dict = {}
        table = pd.read_csv(csv, index_col=0)
        drone_time_list = list(table.loc["time"])[:5]
        drone_time_list.remove(max(drone_time_list))
        drone_time_list.remove(min(drone_time_list))

        drone_time_ave = sum(drone_time_list)/len(drone_time_list)
        time_ave_dict["drone"] = drone_time_ave

        time_height_dict[height] = time_ave_dict
    print(time_height_dict)

    for csv in satellite_list:
        height = csv.split("_")[-2]
        time_ave_dict = {}
        table = pd.read_csv(csv, index_col=0)

        satellite_time_list = list(table.loc["time"][5:10])
        satellite_time_list.remove(max(satellite_time_list))
        satellite_time_list.remove(min(satellite_time_list))
        satellite_time_ave = sum(satellite_time_list)/len(satellite_time_list)
        time_ave_dict["satellite"] = satellite_time_ave
        time_height_dict[height]["satellite"] = time_ave_dict["satellite"]
    return time_height_dict


if __name__ == '__main__':
    model_list = ["vgg", "resnet", "resnest", "seresnet", "cbamresnet", "dense", "efficientv1", "inception"]
    path = "result"
    forming_precision_table(model_list, path)
    # print(select_best_weight("resnet"))
    # print(evaluate_stability("vgg", "recall@1"))
    # adaption = evaluate_adaption_rate("resnet")
    # print(adaption)
    # print(evaluate_realtime("vgg"))
