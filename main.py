from PIL import Image
import cv2
import time, json
import myclip.clip3 as myclip


SOURCE = 0
LANGUAGE = 'cn'
DATASET_PATH = "./datasets/garbage2"
LABEL_PATH = "./labels/labels_en.json"
SELECT_PATH = "./labels/labels_select.json"


def loadLables():
    global labels

    with open(LABEL_PATH, "r") as f:
        label_data = json.load(f)

    with open(SELECT_PATH, "r", encoding="utf-8") as f:
        select_data = json.load(f)

    labels = []
    for idx, ((key1, value1), (key2, value2)) in enumerate(zip(label_data.items(), select_data.items())):
        assert key1 == key2
        if "_" not in value2:
            continue
        name_en = value1
        name_cn = value2.split("_")[-1]
        class_name = value2.split("_")[0]
        label = (idx, name_en, name_cn, class_name)

        labels.append(label)


def delLine(length):
    print("\b" * length * 2, end='\r')
    print(" " * length * 2, end='\r')
    print("\b" * length * 2, end='\r')


def main():
    loadLables()

    if LANGUAGE.lower() == 'en':
        obj_names = [x[1] for x in labels]
    if LANGUAGE.lower() == 'cn':
        obj_names = [x[2] for x in labels]
    myclip.setup(obj_names)

    cap = cv2.VideoCapture(SOURCE)

    t = time.time()
    line = ""
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Video', frame)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # img = Image.fromarray(frame)
            max_i, max_p = myclip.predict(img)
            idx, name_en, name_cn, class_name = labels[max_i]

            delLine(len(line))
            line = f"[{1/(time.time()-t):.0f}fps]\t{max_p*100:.0f}%\t{name_cn}\t{class_name}"
            t = time.time()
            print(line, end='\r')
        
        # 等待用户按下ESC键退出
        if cv2.waitKey(1) == 27:
            break

    cap.release() # 释放摄像头资源
    cv2.destroyAllWindows() # 关闭所有窗口


if __name__ == '__main__':
    main()
