"""
Huy Anh Nguyen, CS Stony Brook University
Created on Mar 28, 2023 - 8:34 PM, EDT

Convert DAVIS format to YouTubeVOS for VISOR dataset
TODO:
    [] Create GroundTruths and JPEGImages folder. Copy from DAVIS
    [] Monitor if a new object appear -> add to Annotations folder
"""
import os
import shutil
from PIL import Image
import numpy as np

path = './Hand_VISOR_Val_v2/Trimmed_VISOR_2022/Annotations/480p'
dst_path = './Trimmed_VISOR_Val_YTVOS/Annotations'

for subseq in os.listdir(path):
    os.makedirs(os.path.join(dst_path, subseq))
    stored_obj = set()
    for frame in sorted(os.listdir(os.path.join(path, subseq))):
        mask = np.array(Image.open(os.path.join(path, subseq, frame)).convert('P'))
        objs = np.unique(mask)
        copy = False
        for obj in objs:
            if obj not in stored_obj:
                stored_obj.add(obj)
                copy = True
        
        if copy:
            src = os.path.join(path, subseq, frame)
            dst = os.path.join(dst_path, subseq, frame)
            shutil.copy(src, dst)




