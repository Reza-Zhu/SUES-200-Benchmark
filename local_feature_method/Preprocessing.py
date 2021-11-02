import glob
import os
import re

'''
type:
In Testing:

query_drone
query_satellite
gallery_drone
gallery_satellite

'''

def get_datasets_list(height, type):
    datasets_path = ".." + os.sep + ".." + os.sep + "Datasets"
    test_path = os.path.join(datasets_path, "Testing", str(height), type)
    test_list = glob.glob(os.path.join(test_path, "*"))
    return sorted(test_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))


if __name__ == '__main__':
    print(get_datasets_list(150, "query_drone"))
