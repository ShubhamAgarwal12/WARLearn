import numpy as np
import os
import cv2
import math
import random

# only use the image including the labeled instance objects for training
def load_annotations(annot_path):
    print(annot_path)
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt]
    return annotations


# print('*****************Add haze offline***************************')
def parse_annotation(annotation):

    line = annotation.split()
    image_path = line[0]
    # print(image_path)
    img_name = image_path.split('/')[-1]
    # print(img_name)
    image_name = img_name.split('.')[0]
    # print(image_name)
    image_name_index = img_name.split('.')[1]
    # print(image_name_index)

#'/data/vdd/liuwenyu/data_vocfog/train/JPEGImages/'
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = cv2.imread(image_path)
    for i in range(10):
        def AddHaz_loop(img_f, center, size, beta, A):
            (row, col, chs) = img_f.shape

            for j in range(row):
                for l in range(col):
                    d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                    td = math.exp(-beta * d)
                    img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
            return img_f

        img_f = image/255
        (row, col, chs) = image.shape
        A = 0.5  
        # beta = 0.08  
        beta = 0.01 * i + 0.05

        img_name = 'data/VOC0712/images/test_foggy/' + image_name \
                   + '__' + ("%.2f"%beta) + '.' + image_name_index
        if os.path.isfile(img_name):
            continue

        size = math.sqrt(max(row, col)) 
        center = (row // 2, col // 2)  
        foggy_image = AddHaz_loop(img_f, center, size, beta, A)
        img_f = np.clip(foggy_image*255, 0, 255)
        img_f = img_f.astype(np.uint8)
        print(img_name)
        cv2.imwrite(img_name, img_f)


if __name__ == '__main__':
    an = load_annotations('data/VOC0712/test.txt')
    ll = len(an)
    print(ll)
    for j in range(ll):
        print(j)
        parse_annotation(an[j])

