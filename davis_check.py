import os
from PIL import Image
import numpy as np
from tqdm import tqdm

vidsPath = './VISOR_2022/JPEGImages/480p'
annsPath = './VISOR_2022/Annotations/480p'
videos = sorted(os.listdir(vidsPath))
for vid in tqdm(videos):
    frames = sorted(os.listdir(os.path.join(vidsPath, vid)))
    masks = sorted(os.listdir(os.path.join(annsPath, vid)))
    assert len(frames) == len(masks), 'Number of frames is different from number of masks for the video: {}'.format(vid)
    assert len(frames) == int(frames[-1].split('.')[0]) + 1, 'Looks like the frame does not end with proper number'
    assert int(frames[0].split('.')[0]) == 0, 'Frame does not start with 0'
    assert len(masks) == int(masks[-1].split('.')[0]) + 1, 'Looks like the mask does not end with proper number'
    assert int(masks[0].split('.')[0]) == 0, 'Mask does not start with 0'
    allMasks = []
    for frm in frames:
        mskPath = os.path.join(annsPath, vid, frm.split('.')[0] + '.png')
        msk = np.array(Image.open(mskPath))
        allMasks.append(msk)
    allMasks = np.stack(allMasks, axis=0)
    uniqueVals = np.unique(allMasks).tolist()
    maxVal = np.max(allMasks)
    if (maxVal + 1) != len(uniqueVals):
        print(uniqueVals)
        assert maxVal + 1 == len(uniqueVals), 'Looks like the object masks are not continuous'

print("PASSED ALL TESTS")
