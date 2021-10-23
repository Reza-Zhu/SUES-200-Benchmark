import os
import shutil

video_name = ["150", "200", "250", "300"]
path = ".." + os.sep + ".." + os.sep + "Datasets" + os.sep + "Training"
if not os.path.exists(path):
    os.mkdir(path)

for name in video_name:
    dst_path = os.path.join(path,name)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for i in range(8):
        frame_path = ".." + os.sep + ".." + os.sep + "Datasets" + os.sep + "drone_view_cut_frame"
        dir_name = "{:0>4d}".format(i + 1)
        frame_num_path = os.path.join(frame_path, dir_name)
        # print(frame_num_path)
        frame_num_name_path = os.path.join(frame_num_path, name)
        dst_name_path = os.path.join(dst_path,dir_name)
        print(dst_name_path)
        # print(frame_num_name_path)
        shutil.copytree(frame_num_name_path,dst_name_path)
        #
            # if not os.path.exists(frame_path):



