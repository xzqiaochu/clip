import numpy as np
import cv2
from PIL import ImageGrab

while(True):
    # 获取屏幕截图
    img = ImageGrab.grab()
    # 转换成OpenCV可处理的格式
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    # 显示图像
    cv2.imshow("screen", frame)
    # 按 'q' 退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放窗口资源
cv2.destroyAllWindows()
