# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
from os import getcwd
import json


root_path = '/data/kb/zhaoshihao/DATA/'

sets = ['train', 'val', 'test']
# classes = ['face', 'normal', 'phone', 'write',
#            'smoke', 'eat', 'computer', 'sleep']

classPath = root_path +'/TEA_SINGLE_OVERWIEW_5_v0.5/voc_tea_classes.json'
classJson = json.load(open(classPath))
classes = list(classJson.keys())
print(classes)
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    # try:
        in_file = open(root_path + 'TEA_SINGLE_OVERWIEW_5_v0.5/Annotations/%s.xml' % (image_id), encoding='utf-8')
        out_file = open(root_path + 'TEA_SINGLE_OVERWIEW_5_v0.5/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                           " ".join([str(a) for a in bb]) + '\n')
    # except Exception as e:
    #     print(e, image_id)


wd = getcwd()
for image_set in sets:
    if not os.path.exists(root_path + 'TEA_SINGLE_OVERWIEW_5_v0.5/labels/'):
        os.makedirs(root_path + 'TEA_SINGLE_OVERWIEW_5_v0.5/labels/')
    image_ids = open(root_path + 'TEA_SINGLE_OVERWIEW_5_v0.5/labels/%s.txt' %
                     (image_set)).read().strip().split()
    list_file = open(root_path + 'TEA_SINGLE_OVERWIEW_5_v0.5/%s.txt' % (image_set), 'w')
    for image_id in tqdm(image_ids):
        imgDir1 = root_path + 'TEA_SINGLE_OVERWIEW_5_v0.5/images/%s.jpg\n' % (image_id)
        imgDir1 = imgDir1[:-1]
        imgDir2 = root_path + 'TEA_SINGLE_OVERWIEW_5_v0.5/images/%s.JPG\n' % (image_id)
        imgDir2 = imgDir2[:-1]
        if os.path.exists(imgDir1):
            list_file.write(root_path + 'TEA_SINGLE_OVERWIEW_5_v0.5/images/%s.jpg\n' % (image_id))
        elif os.path.exists(imgDir2):
            list_file.write(root_path + 'TEA_SINGLE_OVERWIEW_5_v0.5/images/%s.JPG\n' % (image_id))
        else:
            continue

        convert_annotation(image_id)
    list_file.close()
