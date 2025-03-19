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
def parse_annotation(annotation, betas):

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
        def lowlight_loop(img_f, lowlight_param):
            (row, col, chs) = img_f.shape
            img = np.power(img_f, lowlight_param)
            return img

        img_f = image/255
        (row, col, chs) = image.shape
        A = 0.5  
        # beta = 0.08  
        beta = round(random.uniform(1.5, 5),2)
        while(beta in betas):
            beta = round(random.uniform(1.5, 5),2)
        betas.append(beta)
        size = math.sqrt(max(row, col)) 
        center = (row // 2, col // 2)  
        lowlight_image = lowlight_loop(img_f, beta)
        img_f = np.clip(lowlight_image*255, 0, 255)
        img_f = img_f.astype(np.uint8)
        print(img_name)
        img_name = 'data/VOC0712/images/train_lowlight/' + image_name \
                   + '__' + ("%.2f"%beta) + '.' + image_name_index
        cv2.imwrite(img_name, img_f)


if __name__ == '__main__':
    an = load_annotations('data/VOC0712/train.txt')
    ll = len(an)
    print(ll)
    for j in range(ll):
        print(j)
        betas = []
        parse_annotation(an[j], betas)

