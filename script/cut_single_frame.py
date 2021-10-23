import os
import cv2

path = '../../Datasets/drone-view/0001/150/150-0.jpg'
img = cv2.imread(path)
x, y, _ = img.shape
cut_size = (y - 1080) / 2
img = img[:, int(cut_size):int(y - cut_size)]
cv2.imwrite("img.jpg", img)
