import numpy as np
import torch
from utils import get_yaml_value
import sys
import os
import yaml
# os.system("source /home/ubuntu/pytorch/bin/activate && python ppt.py")
# os.system("")
# dict_name = {"batch_size": 10}
with open("test.yaml", "r", encoding="utf-8") as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)
        dict['batch_size'] = 10
        with open("test.yaml", "w", encoding="utf-8") as f:
                yaml.dump(dict, f)
