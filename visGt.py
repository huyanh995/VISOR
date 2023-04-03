# Visualize groundtruth for videos containing person and object trajectories
import os
import json
import random
from collections import defaultdict
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from colors import sampleColor


def drawBbox(im, box, color):
    left, top, right, bottom = box
    cv2.line(im, (left, top), (right, top), color, thickness=3)
    cv2.line(im, (right, top), (right, bottom), color, thickness=3)
    cv2.line(im, (right, bottom), (left, bottom), color, thickness=3)
    cv2.line(im, (left, bottom), (left, top), color, thickness=3)
    return im


vidsPath = './VISOR_2022/JPEGImages/480p'
annsPath = './VISOR_2022/Annotations/480p'
videos = os.listdir(vidsPath)
# random.shuffle(videos)
# videos = videos[:50]

for vid in tqdm(videos):
    os.makedirs('./GTVisualizations/' + vid)
    frames = sorted(os.listdir(os.path.join(vidsPath, vid)))
    masks = sorted(os.listdir(os.path.join(annsPath, vid)))
    for frm in frames:
        frmPath = os.path.join(vidsPath, vid, frm)
        im = cv2.imread(frmPath)
        mskPath = os.path.join(annsPath, vid, frm.split('.')[0] + '.png')
        msk = np.array(Image.open(mskPath))
        uniqueVals = np.unique(msk).tolist()
        for idx, val in enumerate(uniqueVals):
            if val == 0: continue
            color = sampleColor(val)
            row, col = np.where(msk == val)
            ymin, ymax = np.min(row), np.max(row)
            xmin, xmax = np.min(col), np.max(col)
            box = [xmin, ymin, xmax, ymax]
            box = [int(b) for b in box]
            im = drawBbox(im, box, color)
        cv2.imwrite(os.path.join('./GTVisualizations/', vid, frm), im)
