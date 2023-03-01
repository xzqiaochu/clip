import cv2

cap = cv2.VideoCapture(1)

while True:
    # 读取并丢弃缓冲区中的帧，直到只剩下一帧为止
    # while cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1:
    #     ret, frame = cap.read()

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
