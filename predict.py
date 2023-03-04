from PIL import Image
import cv2

import myclip.clip3trt as myclip
from myclip.utils import *


SOURCE = 0
LABEL_CSV = "./labels/2022.csv"


def main():
    labels = loadLabels(LABEL_CSV)

    myclip.setup(labels)

    cap = cv2.VideoCapture(SOURCE)
    
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera', frame)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            max_i, max_p = myclip.predict(img)
            category, cn_name = labels[max_i][0:2]

            buff = f"\r[{getFPS():2.0f}fps]\t{max_p*100:3.0f}%\t{category}\t{cn_name}"
            buff += " " * 20
            print(buff, end='')
        
        # 等待用户按下ESC键退出
        if cv2.waitKey(1) == 27:
            break

    cap.release() # 释放摄像头资源
    cv2.destroyAllWindows() # 关闭所有窗口


if __name__ == '__main__':
    main()
