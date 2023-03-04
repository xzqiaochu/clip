import time
import csv


last_t = 0


def getFPS():
    global last_t
    fps = 1 / (time.time() - last_t)
    last_t = time.time()
    return fps


def loadLabels(csv_file):
    labels = []
    with open(csv_file, 'r', encoding='gbk') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                labels.append(row)
    return labels