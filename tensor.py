import numpy as np
import torch

index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
good_index = [[1],
              [2]]
mask = np.in1d(index, good_index)
# print(mask)
# good = np.argwhere(mask == True)
# print(good)

from model_ import model_dict

model1 = model_dict['resnet'](100, 0.4)
model2 = model_dict['vgg'](100, 0.5)
print(model1,model2)