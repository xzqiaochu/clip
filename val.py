import os, time, random
from PIL import Image
import csv

import myclip.clip3trt as myclip


NUM = 1000
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

    precise_cnt = 0
    correct_cnt = 0
    wrong_cnt = 0
    for i in range(NUM):
        idx, category, cn_name, en_name = random.choice(labels) # 随机选择一个类别
        imgs_path = os.path.join(DATASET_PATH, str(idx)) # 类别的路径
        img_path = os.path.join(imgs_path, random.choice(os.listdir(imgs_path))) # 随机选择该类别里的一个图片
        img = Image.open(img_path) # 读入图片

        max_i, max_p = myclip.predict(img)
        p_idx, p_category, p_cn_name, p_en_name = labels[max_i]

        if idx == p_idx:
            precise_cnt += 1
            add_output = "precise"
        elif category == p_category:
            correct_cnt += 1
            add_output = "correct"
        else:
            wrong_cnt += 1
            add_output = f"wrong\t{category}({cn_name})\t->\t{p_category}({p_cn_name})\t{img_path}"

        print(f"\r[{getFPS():.0f}fps]\t{i}\t{max_p*100:.0f}%\t{add_output}")

    total = precise_cnt + correct_cnt + wrong_cnt
    print("")
    print("total:\t%d" % total)
    print("precise:\t%d(%.1f%%)" % (precise_cnt, precise_cnt / total * 100))
    print("correct:\t%d(%.1f%%)" % (correct_cnt, correct_cnt / total * 100))
    print("wrong:\t%d(%.1f%%)" % (wrong_cnt, wrong_cnt / total * 100))


if __name__ == "__main__":
    main()
