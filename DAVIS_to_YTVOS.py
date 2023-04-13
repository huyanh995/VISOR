"""
Huy Anh Nguyen, CS Stony Brook University
Created on Mar 28, 2023 - 8:34 PM, EDT

Convert DAVIS format to YouTubeVOS for VISOR dataset
TODO:
    [] Adding multi-processing to speed up
"""
import os
import shutil
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm


def main(in_path: str, out_path: str) -> None:
    # Check how many set in VISOR
    dataset = os.listdir(os.path.join(in_path, 'ImageSets', '2022'))
    src_img = os.path.join(in_path, 'JPEGImages', '480p')
    src_anno = os.path.join(in_path, 'Annotations', '480p')

    for s in dataset:
        print(f'Moving {s} set...')
        # Making a folder in YTVOS format
        dst_path = os.path.join(out_path, s.replace('.txt', ''))
        dst_img = os.path.join(dst_path, 'JPEGImages')
        dst_anno = os.path.join(dst_path, 'Annotations')
        dst_gt = os.path.join(dst_path, 'GTMasks')

        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_anno, exist_ok=True)
        os.makedirs(dst_gt, exist_ok=True)

        # Read list of subseq in s split
        with open(os.path.join(in_path, 'ImageSets', '2022', s), 'r') as f:
            ls_subseq = [i.strip() for i in list(f.readlines())]

        for subseq in tqdm(ls_subseq):
            os.makedirs(os.path.join(dst_img, subseq))
            os.makedirs(os.path.join(dst_anno, subseq))
            os.makedirs(os.path.join(dst_gt, subseq))

            stored_obj = set()
            for frame in sorted(os.listdir(os.path.join(src_anno, subseq))):
                # Always copy GTMask and Images
                img_name = frame.replace('.png', '.jpg')
                shutil.copy(os.path.join(src_img, subseq, img_name),
                            os.path.join(dst_img, subseq, img_name))

                shutil.copy(os.path.join(src_anno, subseq, frame),
                            os.path.join(dst_gt, subseq, frame))

                mask = np.array(Image.open(os.path.join(
                    src_anno, subseq, frame)).convert('P'))
                objs = np.unique(mask)
                copy = False

                # Scan for new object in masks -> Copy only it.
                for obj in objs:
                    if obj == 0:
                        continue  # Ignore background
                    if obj not in stored_obj:
                        stored_obj.add(obj)
                        copy = True

                if copy:
                    shutil.copy(os.path.join(src_anno, subseq, frame),
                                os.path.join(dst_anno, subseq, frame))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='Input VISOR in DAVIS format dataset', default='../Hand_n_objects/VISOR_2022')
    parser.add_argument(
        '-o', '--output', help='Output directory', default='../Hand_n_objects')
    args = parser.parse_args()

    out_path = os.path.join(args.output, 'YTVOS_VISOR_2022')
    main(args.input, out_path)
