import hololive_categorizer
import torch
import glob
import random
import cv2

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def CropPerson(PATH):
    files = glob.glob(PATH)
    i = 0
    for file in files:
        print(file)
        img = cv2.imread(file, 1)
        results = yolo_model(img)
        df = results.pandas().xyxy[0]
        if df.empty:
            print('DataFrame is empty')
        else:
            for index, row in df.iterrows():
                if row.iloc[5] == 0:
                    h, w, c = img.shape
                    new_img = img[(h - int(row.ymax)): (h - int(row.ymin)),
                                  int(row.xmin): int(row.xmax)]
                    cv2.imwrite('./Train_Data/testcrop/' + str(i)+'.jpg',
                                new_img)
                    i = i + 1


def ShowPersonDetect():
    files = glob.glob(hololive_categorizer.PATH_PEKORA)
    file = random.choice(files)
    print(file)
    img = cv2.imread(file, 1)
    cv2.imshow('original image', img)

    results = yolo_model(img)

    df = results.pandas().xyxy[0]

    if df.empty:
        print('DataFrame is empty')
    else:
        print(df)
        print(type(df.name))
        for index, row in df.iterrows():
            if row.iloc[5] == 0:
                h, w, c = img.shape
                new_img = img[(h - int(row.ymax)): (h - int(row.ymin)),
                              int(row.xmin): int(row.xmax)]
                cv2.imshow('reshaped image' + str(index), new_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    CropPerson(hololive_categorizer.PATH_PEKORA)
