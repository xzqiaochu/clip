import time
from PIL import Image
import cv2
import csv

import myclip.clip3 as myclip


SOURCE = 1
LANGUAGE = "cn"
LABEL_CSV = "./labels/labels.csv"
DATASET_PATH = "./datasets/garbage2"

last_t = 0


def getFPS():
    global last_t
    fps = 1 / (time.time() - last_t)
    last_t = time.time()
    return fps


def main():
    labels = []
    with open(LABEL_CSV, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(row)

    if LANGUAGE.lower() == 'cn':
        labels_t = [x[2] for x in labels]
    elif LANGUAGE.lower() == 'en':
        labels_t = [x[3] for x in labels]

    myclip.setup(labels_t)

    cap = cv2.VideoCapture(SOURCE)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera', frame)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            max_i, max_p = myclip.predict(img)
            idx, category, cn_name, en_name = labels[max_i]

            buff = f"\r[{getFPS():.0f}fps]\t{max_p*100:.0f}%\t{category}\t{cn_name}"
            buff += " " * 20
            print(buff, end='')
        
        # 等待用户按下ESC键退出
        if cv2.waitKey(1) == 27:
            break

    cap.release() # 释放摄像头资源
    cv2.destroyAllWindows() # 关闭所有窗口


if __name__ == '__main__':
    main()
