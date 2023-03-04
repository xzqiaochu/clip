import cv2
import time

cap = cv2.VideoCapture(0)

while True:
    t = time.time()
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Video', frame)
        print(f"\r[{1/(time.time()-t):2.0f}fps]", end="")
        t = time.time()

    # 等待用户按下ESC键退出
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
