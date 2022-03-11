import cv2
import os
import glob

video_name = [150, 200, 250, 300]
NUMBER = 50

for k in range(3):
    dir_str = "{:0>4d}".format(k+1)
    drone_view_path = os.path.join(".." + os.sep + "..", "drone-view", dir_str)
    print(drone_view_path)
    datasets_path = os.path.join(".." + os.sep + "..", "Datasets", "drone-view", dir_str)
    print(datasets_path)
    if not os.path.exists(datasets_path):
        os.mkdir(datasets_path)
    video_list = glob.glob(os.path.join(drone_view_path, "*"))
    frame_count = []
    print(video_list)
    for video in video_list:
        count = 0
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()
            count = count + 1
            if not ret:
                break
            # cv2.imshow("img",frame)
            # cv2.waitKey(1)
        frame_count.append(count)
        print(frame_count)
    # frame_count = [3096, 2295, 2045, 2373]
    fig_count = 0
    frame_name = dict(zip(video_list,frame_count))
    for i in frame_name.items():
        if not os.path.exists(os.path.join(datasets_path, str(video_name[fig_count]))):
            os.mkdir(os.path.join(datasets_path, str(video_name[fig_count])))
        num = round(i[1] / NUMBER)
        print("num:", num)
        # for video in video_list:
        count = 0
        num_saved = 0
        cap = cv2.VideoCapture(i[0])
        while cap.isOpened():
            ret, frame = cap.read()
            if count % int(num) == 0:
                cv2.imwrite(os.path.join(datasets_path,
                                         os.path.join(str(video_name[fig_count]),
                                         str(video_name[fig_count])+"-"+str(count)+".jpg")),
                                         frame)
                num_saved += 1
            if not ret or num_saved == 50:
                break
            count += 1
        print(dir_str + "---" + str(video_name[fig_count]), "--saved--totally:", num_saved)
        fig_count += 1
