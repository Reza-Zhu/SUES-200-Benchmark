import os
import time
import yaml
from train_func import train
from test_and_evaluate import main


def parameter(index_name, index_number):
    with open("settings.yaml", "r", encoding="utf-8") as f:
        setting_dict = yaml.load(f, Loader=yaml.FullLoader)
        setting_dict[index_name] = index_number
        f.close()
        with open("settings.yaml", "w", encoding="utf-8") as f:
            yaml.dump(setting_dict, f)
            f.close()



def Auto_tune(model_list,  height_list, drop_rate, learning_rate):
    for model in model_list:
        parameter("model", model)
        for height in height_list:
            parameter("height", height)
            for dr in drop_rate:
                parameter("drop_rate", dr)
                for lr in learning_rate:
                    parameter("lr", lr)
                    # for wd in weight_decay:
                    #     parameter("weight_decay", wd)
                    with open("settings.yaml", "r", encoding="utf-8") as f:
                        setting_dict = yaml.load(f, Loader=yaml.FullLoader)
                        print(setting_dict)
                        f.close()
                    train()
                    main()

height_list = [150, 200, 250, 300]
learning_rate = [0.005, 0.01, 0.015, 0.02]
drop_rate = [0.2, 0.3, 0.4]
# weight_decay = [0.001, 0.0001]
model_list = ["efficientv2"]
# "vgg", "resnet", "seresnet", "resnest","cbamresnet"
Auto_tune(model_list, height_list, drop_rate, learning_rate)

