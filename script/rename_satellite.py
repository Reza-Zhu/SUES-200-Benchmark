import glob
import os
import re


raw_datasets_path = "/media/data1/RAW_DATASETS"
satellite_data_path = os.path.join(raw_datasets_path, "satellite-view")
satellite_data_list = glob.glob(os.path.join(satellite_data_path, "*"))

for i in satellite_data_list:
    imgs = glob.glob(os.path.join(i, "*"))
    for j in imgs:
        # print(os.path.dirname(j))
        dst = os.path.join(os.path.dirname(j), "0.png")
        print(dst)
        os.rename(j, dst)


