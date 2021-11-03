import glob
import os
import re


raw_datasets_path = "../../RAW_DATASETS"
satellite_data_path = os.path.join(raw_datasets_path, "satellite-view")
satellite_data_list = glob.glob(os.path.join(satellite_data_path, "*"))

for i in satellite_data_list:
    imgs = glob.glob(os.path.join(i, "*"))
    for j in imgs:
        dst = j[:39] + "0.png"
        # print(src)
        os.rename(j,dst)
        # print(j[:39])


