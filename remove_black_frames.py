"""
Huy Anh Nguyen, CS Stony Brook University
Created on Mar 28, 2023 - 0:10 PM, EDT

Remove black frames from VISOR dataset in DAVIS format
"""

import os
import argparse
import json

from PIL import Image
import numpy as np
from tqdm import tqdm

# Loop through Annotations folder
def main(in_path: str, remove_trail: bool) -> None:
    # in_path should be a directory in DAVIS format
    with open(os.path.join(in_path, 'hand_relation.json'), 'r') as f:
        hand_relation = json.load(f)
    num_removed = 0
    for res_path in os.listdir(os.path.join(in_path, 'Annotations')):
        # VISOR > Annotations > resolution > subsquences
        data_path = os.path.join(in_path, 'Annotations', res_path)
        for subseq_path in tqdm(os.listdir(data_path)):
            # Forward pass
            change = False
            for mask_name in sorted(os.listdir(os.path.join(data_path, subseq_path))):
                mask_path = os.path.join(data_path, subseq_path, mask_name)
                img_path = os.path.join(data_path.replace('Annotations', 'JPEGImages'), subseq_path, mask_name.replace('.png', '.jpg'))
                mask = np.array(Image.open(mask_path).convert('P'))
                if len(np.unique(mask)) == 1:
                    change = True
                    os.remove(mask_path)
                    os.remove(img_path)
                    # delete in hand_relation
                    del hand_relation[subseq_path][mask_name.replace('.png', '.jpg')]
                    num_removed += 1
                    
                else:
                    break # stop looping whenever met a non black mask
            
            # Backward pass
            if remove_trail: # Not all experiments need to remove trail black frames
                for mask_name in sorted(os.listdir(os.path.join(data_path, subseq_path)), reverse=True):
                    mask_path = os.path.join(data_path, subseq_path, mask_name)
                    img_path = os.path.join(data_path.replace('Annotations', 'JPEGImages'), subseq_path, mask_name.replace('.png', '.jpg'))
                    mask = np.array(Image.open(mask_path).convert('P'))
                    if len(np.unique(mask)) == 1:
                        change = True
                        os.remove(mask_path)
                        os.remove(img_path)
                        # delete in hand_relation
                        del hand_relation[subseq_path][mask_name.replace('.png', '.jpg')]
                        num_removed += 1
                        
                    else:
                        break # stop looping whenever met a non black mask
                
            # Change name
            if change:
                list_frames = sorted(os.listdir(os.path.join(data_path, subseq_path))) # Get new list frames after deleting
                
                for idx, frame in enumerate(list_frames):
                    mask_src_path = os.path.join(data_path, subseq_path, frame)
                    mask_dst_path = os.path.join(data_path, subseq_path, f'{idx:05}.png')
                    img_src_path = os.path.join(data_path.replace('Annotations', 'JPEGImages'), subseq_path, frame.replace('.png', '.jpg'))
                    img_dst_path = os.path.join(data_path.replace('Annotations', 'JPEGImages'), subseq_path, f'{idx:05}.jpg')
                    os.rename(mask_src_path, mask_dst_path)
                    os.rename(img_src_path, img_dst_path)
                    hand_relation[subseq_path][f'{idx:05}.jpg'] = hand_relation[subseq_path].pop(frame.replace('.png', '.jpg'))
    with open(os.path.join(in_path, 'hand_relation.json'), 'w') as f:
        json.dump(hand_relation, f)
    print(f"Removed {num_removed} black frames")

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parameters for remove black frames from VISOR DAVIS")
    parser.add_argument('-i', '--input', type=str, help="input path to DAVIS formatted dataset", required=True, default='./VISOR_2022')
    parser.add_argument('-trail', action='store_true', help="remove trail black frames")
    args = parser.parse_args()
    input_path = args.input
    remove_trail = args.trail
    
    # input_path = './val/VISOR_2022'
    # remove_trail = False
    
    print(f"CONFIG: path: {input_path}, remove trail black frames {remove_trail}")
    main(input_path, remove_trail)
