import cv2

image = "./imgs/img1.png"
image = cv2.imread(image)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(image, None)
print(des.shape)
