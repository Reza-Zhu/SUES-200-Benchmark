import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x/sum_exp_x

    return y


def select_best_weight(model_name, csv_path):
    # csv_path = "./save_model_weight"
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


def evaluate_adaption_rate(model_name, csv_path):
    drone_list, satellite_list = select_best_weight(model_name, csv_path)
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

    return evaluate_satellite_height / evaluate_drone_height


def forming_precision_table(model_list, save_dir, csv_path):
    for model_name in model_list:
        drone_list, satellite_list = select_best_weight(model_name, csv_path)
        drone_total_frame = pd.DataFrame()
        satellite_total_frame = pd.DataFrame()
        for csv in drone_list:
            table = pd.read_csv(csv, index_col=0)
            query_number = len(list(filter(lambda x: "drone" in x, table.columns))) - 1
            values = list(table.loc["recall@1", :])[:query_number]
            indexes = list(table.loc["recall@1", :].index)[:query_number]
            net_name = indexes[values.index(max(values))]
            recall_ap_list = table.loc[:, net_name].drop(["recall@1p"])
            drone_total_frame = pd.concat([drone_total_frame, recall_ap_list], axis=1)

        drone_total_frame.columns = ["150", "200", "250", "300"]
        drone_total_frame = drone_total_frame.T
        drone_total_frame.to_csv(os.path.join(save_dir, model_name + "_drone.csv"))
        print(drone_total_frame)

        for csv in satellite_list:
            table = pd.read_csv(csv, index_col=0)
            query_number = len(list(filter(lambda x: "drone" in x, table.columns))) - 1
            values = list(table.loc["recall@1", :])[query_number:query_number*2]
            indexes = list(table.loc["recall@1", :].index)[query_number:query_number*2]
            net_name = indexes[values.index(max(values))]
            recall_ap_list = table.loc[:, net_name].drop(["recall@1p"])
            satellite_total_frame = pd.concat([satellite_total_frame, recall_ap_list], axis=1)

        satellite_total_frame.columns = ["150", "200", "250", "300"]
        satellite_total_frame = satellite_total_frame.T
        satellite_total_frame.to_csv(os.path.join(save_dir, model_name + "_satellite.csv"))
        print(satellite_total_frame)


def evaluate_stability(model_list, evaluation_value, csv_path):
    # print(model_name)
    satellite_mae_list = []
    satellite_mean_list = []
    satellite_total = []
    drone_mae_list = []
    drone_mean_list = []
    drone_total = []
    for model_name in model_list:
        drone_list, satellite_list = select_best_weight(model_name, csv_path)
        evaluate_value_drone_height = []
        evaluate_value_satellite_height = []
        for csv in drone_list:
            for height in ["150", "200", "250", "300"]:
                if height in csv:
                    table = pd.read_csv(csv, index_col=0)
                    evaluate_value_drone_height.append(table.at[evaluation_value, "drone_max"])

        mean_drone = np.ones(4) * sum(evaluate_value_drone_height) / 4
        # print(mean_satellite)
        reciprocal_stability_drone = 1 / mean_absolute_error(evaluate_value_drone_height, mean_drone)
        drone_mae_list.append(reciprocal_stability_drone)
        drone_mean_list.append(evaluate_value_drone_height)

        for csv in satellite_list:
            for height in ["150", "200", "250", "300"]:
                if height in csv:
                    table = pd.read_csv(csv, index_col=0)
                    evaluate_value_satellite_height.append(table.at[evaluation_value, "satellite_max"])

        mean_satellite = np.ones(4) * sum(evaluate_value_satellite_height) / 4
        # print(mean_satellite)
        reciprocal_stability_satellite = 1 / mean_absolute_error(evaluate_value_satellite_height, mean_satellite)
        satellite_mae_list.append(reciprocal_stability_satellite)
        satellite_mean_list.append(evaluate_value_satellite_height)

    scale = MinMaxScaler(feature_range=(0.4, 0.6))
    drone_mean_list = np.array(drone_mean_list).reshape(-1, 4)
    drone_mae_list = scale.fit_transform(np.array(drone_mae_list).reshape(-1, 1)).reshape(-1, 1)
    print(drone_mean_list)
    print(drone_mae_list)

    for i in range(len(drone_mean_list)):
        drone_mean_list[i] = np.mean(drone_mean_list[i] * drone_mae_list[i])
        drone_total.append(drone_mean_list[i][0])

    scale = MinMaxScaler(feature_range=(0.4, 0.6))
    satellite_mean_list = np.array(satellite_mean_list).reshape(-1, 4)
    satellite_mae_list = scale.fit_transform(np.array(satellite_mae_list).reshape(-1, 1)).reshape(-1, 1)

    for i in range(len(satellite_mean_list)):
        satellite_mean_list[i] = np.mean(satellite_mean_list[i] * satellite_mae_list[i])
        satellite_total.append(satellite_mean_list[i][0])

    table = pd.DataFrame()

    table["drone"] = drone_total
    table["satellite"] = satellite_total
    table = table.T
    table.columns = model_list

    table.to_csv("result/stability.csv")
    # print(table)
    # return table


def evaluate_realtime(model_name, csv_path):
    drone_list, satellite_list = select_best_weight(model_name, csv_path)
    time_ave_dict = {"drone": 0, "satellite": 0}

    for csv in drone_list:
        height = csv.split("_")[-2]
        table = pd.read_csv(csv, index_col=0)
        query_number = len(list(filter(lambda x: "drone" in x, table.columns))) - 1

        drone_time_list = list(table.loc["time"])[:query_number]
        drone_time_list.remove(max(drone_time_list))
        drone_time_list.remove(min(drone_time_list))

        drone_time_ave = sum(drone_time_list) / len(drone_time_list)
        time_ave_dict["drone"] += drone_time_ave

        # time_height_dict[height] = time_ave_dict

    time_ave_dict["drone"] = time_ave_dict["drone"] / 4
    for csv in satellite_list:
        height = csv.split("_")[-2]
        table = pd.read_csv(csv, index_col=0)
        query_number = len(list(filter(lambda x: "drone" in x, table.columns))) - 1

        satellite_time_list = list(table.loc["time"][query_number:query_number*2])
        satellite_time_list.remove(max(satellite_time_list))
        satellite_time_list.remove(min(satellite_time_list))
        satellite_time_ave = sum(satellite_time_list) / len(satellite_time_list)
        time_ave_dict["satellite"] += satellite_time_ave
    time_ave_dict["satellite"] = time_ave_dict["satellite"] / 4
    return time_ave_dict


if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    model_list = ["vgg", "resnet", "resnest", "seresnet", "cbamresnet", "dense", "efficientv1", "inception"]
    #### stability ####
    csv_path = "/media/data1/save_model_weight"

    evaluate_stability(model_list, "recall@1", csv_path)

    # print("no shared weight")
    # csv_path = "/media/data1/save_model_weight"
    # forming_precision_table(["seresnet"], "result", csv_path)
    #
    # csv_path = "/media/data1/save_datasets_6_weight"
    # print(select_best_weight("seresnet", csv_path))
    # forming_precision_table(["seresnet"], "result", csv_path)

    # print("shared weight")
    # csv_path = "./save_model_weight"
    # forming_precision_table(["seresnet"], "result", csv_path)


    #### ad ####
    # ad = {}
    # for model in model_list:
    #     ad[model] = evaluate_adaption_rate(model)
    #     print(ad)
    # table = pd.DataFrame([ad])
    # table.to_csv("ad.csv")
    # print(table)
    #### time ####
    # time = {}
    # for model in model_list:
    #     time[model] = evaluate_realtime(model)
    #
    # time_table = pd.DataFrame(time)
    # time_table.to_csv("result/real_time.csv")
    # print(time_table)
