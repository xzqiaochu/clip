import cv2

cap = cv2.VideoCapture(0)

while True:

    # 获取一帧视频
    ret, frame = cap.read()

    # 如果成功获取视频帧
    if ret:
        # 进行处理
        # ...

        # 在窗口中实时显示视频帧
        cv2.imshow('Video', frame)

    # 等待用户按下ESC键退出
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头资源
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()
