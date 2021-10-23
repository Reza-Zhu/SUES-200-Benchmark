import cv2
import os
import glob

video_name = ["150", "200", "250", "300"]
sum = 0
for i in range(8):
    cut_frame_dir = "../../Datasets/drone_view_cut_frame"
    if not os.path.exists(cut_frame_dir):
        os.mkdir(cut_frame_dir)
    img_path = ".." + os.sep + ".." + os.sep + "Datasets" + os.sep + "drone-view"
    dir_name = "{:0>4d}".format(i + 1)
    img_path = os.path.join(img_path, dir_name)
    cut_frame_dir = os.path.join(cut_frame_dir, dir_name)
    if not os.path.exists(cut_frame_dir):
        os.mkdir(cut_frame_dir)
    for name in video_name:
        cut_frame_name_dir = cut_frame_dir
        cut_img_path = os.path.join(img_path, name)
        img_list = glob.glob(os.path.join(cut_img_path,"*.jpg"))
        cut_frame_name_dir = os.path.join(cut_frame_name_dir, name)
        if not os.path.exists(cut_frame_name_dir):
            os.mkdir(cut_frame_name_dir)
        count = 0
        for img_ in img_list:
            print(img_+ " is processing")
            img = cv2.imread(img_)
            x, y, _ = img.shape
            cut_size = (y - 1080) / 2
            img = img[:, int(cut_size):int(y - cut_size)]
            cv2.imwrite(os.path.join(cut_frame_name_dir, str(count) + '.jpg'), img)
            count = count+1
            print(os.path.join(cut_frame_name_dir, str(count) + '.jpg'),"  saved")
            sum = sum + 1


print("total images: ", sum)