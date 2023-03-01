from PIL import Image
import os, sys, time, random, json
import myclip.clip3 as myclip


NUM = 1000
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


def main():
    loadLables()
    
    if LANGUAGE.lower() == 'en':
        obj_names = [x[1] for x in labels]
    if LANGUAGE.lower() == 'cn':
        obj_names = [x[2] for x in labels]
    myclip.setup(obj_names)

    precise_cnt = 0
    correct_cnt = 0
    wrong_cnt = 0
    t = time.time()
    for i in range(NUM):
        idx, name_en, name_cn, class_name = random.choice(labels) # 随机选择一个类别
        imgs_path = os.path.join(DATASET_PATH, str(idx)) # 类别的路径
        img_fname = random.choice(os.listdir(imgs_path)) # 随机选择该类别里的一个图片
        img_path = os.path.join(imgs_path, img_fname) # 图片的路径
        img = Image.open(img_path) # 读入图片

        max_i, max_p = myclip.predict(img)
        p_idx, p_name_en, p_name_cn, p_class_name = labels[max_i]

        if idx == p_idx:
            precise_cnt += 1
            add_output = "precise"
        elif class_name == p_class_name:
            correct_cnt += 1
            add_output = "correct"
        else:
            wrong_cnt += 1
            add_output = f"wrong\t{name_cn}({idx})\t{p_name_cn}({p_idx})"

        print(f"[{1/(time.time()-t):.0f}fps]\t{i}\t{img_fname}\t{max_p*100:.0f}%\t{add_output}")
        t = time.time()

    total = precise_cnt + correct_cnt + wrong_cnt
    print("")
    print("total:\t%d" % total)
    print("precise:\t%d(%.1f)" % (precise_cnt, precise_cnt / total * 100))
    print("correct:\t%d(%.1f)" % (correct_cnt, correct_cnt / total * 100))
    print("wrong:\t%d(%.1f)" % (wrong_cnt, wrong_cnt / total * 100))


if __name__ == "__main__":
    main()
