"""
Huy Anh Nguyen, CS Stony Brook University
Created on Mar 14, 2023 - 6:12 PM, EDT

Generate trajectories from VISOR annotation in YouTubeVOS format
where in val set, only mask where a new object appears is added
"""

import json
import os
import glob
import argparse
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image

from tqdm import tqdm
from filter_hand_object import filter_annotation, HAND_IDS, GLOVE_IDS

SPARSE_ANNO_PATH = '../GroundTruth-SparseAnnotations/annotations'
SPARSE_IMG_PATH = '../GroundTruth-SparseAnnotations/rgb_frames'
HANDS_MAPPING = {'left hand': 1,
                'right hand': 2,
                'left glove': 3,
                'right glove': 4,
                'glove': 5,
                }

NUM_ENTITIES = 0
NUM_FRAMES = 0

def is_trajectory(traj: list, window: int = 2) -> bool:
    res = False
    for i in range(window, len(traj) + 1):
        tmp_set = set(traj[i - window : i])
        if len(tmp_set) == 1 and tmp_set != set([None]):
            res = True
    return res


# def is_hand_appear(traj: list) -> bool:
#     set_traj = set(traj)
#     set_traj.discard(None)
#     return len(set_traj) > 0

def process_poly(segments: list) -> list:
    polygons = []
    for poly in segments:
        if poly == []:
            polygons.append([[0.0, 0.0]])
        polygons.append(np.array(poly, dtype = np.int32))
    
    return polygons


def imwrite_davis(out_path: str, img):
    davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
    davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                             [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                             [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                             [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                             [0, 64, 128], [128, 64, 128]]
    assert len(img.shape) < 4 or img.shape[0] == 1 # Only i img
    img = Image.fromarray(img, 'P')
    img.putpalette(davis_palette.ravel())
    img.save(out_path)
    

def gen_davis_mask(subseq_name: str, frames:list, object_to_color: dict, dataset: str, out_path: str, resolution: tuple):
    # Not count hands by default
    h, w = resolution
    anno_path = os.path.join(out_path, 'VISOR_DAVIS', dataset, 'Annotations', subseq_name)
    image_path = os.path.join(out_path, 'VISOR_DAVIS', dataset, 'JPEGImages', subseq_name)
    
    # Create folders for new subsequence
    os.makedirs(anno_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)

    relation_map = {}
    global NUM_FRAMES
    NUM_FRAMES += len(frames)
    
    for idx, frame in enumerate(frames):
        global NUM_ENTITIES
        NUM_ENTITIES += len(frame['annotations'])
        
        frame_name = f'{idx:05}'
        # Generate relation map  
        relation_map[frame_name + '.jpg'] = {}
        relation_map[frame_name + '.jpg']['relations'] = [(HANDS_MAPPING[frame['annotations'][k]['name']], \
                                            object_to_color[frame['annotations'][v]['name']] + 5) \
                                            for k, v in frame['relations']]

        # Generate mask
        img_path = os.path.join(SPARSE_IMG_PATH, dataset, frame['image']['video'].split('_')[0], frame['image']['image_path'])
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)
        
        mask = np.zeros(img.shape[:-1], dtype = np.uint8)
        
        # Get the polygons of hand-held objects from relation annotation
        for hand_id, object_id in frame['relations']:
            object_name = frame['annotations'][object_id]['name']
            
            # polygons = []
            # for poly in frame['annotations'][object_id]['segments']:
            #     if poly == []:
            #         polygons.append([[0.0, 0.0]])
            #     polygons.append(np.array(poly, dtype=np.int32))
            
            polygons = process_poly(frame['annotations'][object_id]['segments'])
            # Create mask using collected polygons
            color = object_to_color[object_name]
            cv2.fillPoly(mask, polygons, (color, color, color))
            
        # Resize mask to match with img
        resized_mask = cv2.resize(mask, resolution, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        # Write annotation mask into Annotations and original image to JPEGImages
        imwrite_davis(os.path.join(anno_path, frame_name + '.png'), resized_mask)
        cv2.imwrite(os.path.join(image_path, frame_name + '.jpg'), resized_img)

    return relation_map

def dilate(mask, iterations = 5):
    mask = ndimage.binary_dilation(mask, iterations = iterations).astype(mask.dtype)
    return mask

def filter_object_segment(hand_segment: list, object_segment: list) -> list:
    # Create hand mask
    hand_poly = process_poly(hand_segment)
    hand_mask = np.zeros((1080, 1920), dtype = np.uint8)
    cv2.fillPoly(hand_mask, hand_poly, (1, 1, 1))
    hand_mask = dilate(hand_mask)
    
    res = []
    for segment in object_segment:
        object_mask = np.zeros((1080, 1920), dtype = np.uint8)
        object_poly = process_poly([segment])
        cv2.fillPoly(object_mask, object_poly, (1, 1, 1))
        # Dilate the object_mask
        object_mask = dilate(object_mask)
        
        # Check for overlapping between hand mask and object mask
        overlap_score = np.sum(np.bitwise_and(hand_mask, object_mask))
        if overlap_score > 0:
            res.append(segment)
            
    return res


def process_subseq(subseq_name: str, list_frames: list, dataset: str, out_path: str, resolution: tuple) -> None:
    h, w = resolution
    # Get list of frames from a subsequence
    frames = sorted(list_frames, key=lambda k: k['image']['image_path'])
    left_traj = [frame['left'] for frame in frames]
    right_traj = [frame['right'] for frame in frames]

    # Check if any trajectory is valid
    if is_trajectory(left_traj) or is_trajectory(right_traj):
        # Get set of objects in a valid subsequence
        # Reserve 1 for left hand, 2 for right hand
        set_objects = set(left_traj + right_traj)
        set_objects.discard(None)

        color_to_object = {k + 1: obj for k, obj in enumerate(sorted(set_objects))} # Try to get some order
        object_to_color = {v: k for k, v in color_to_object.items()}
        
        # TODO: Filter only nearby component of objects. Need to modify frames
        for idx in range(len(frames)):
            updated_annotations = {}
            frame = frames[idx]
            for (hand_id, object_id) in frame['relations']:
                hand_segments = frame['annotations'][hand_id]['segments']
                object_segments = frame['annotations'][object_id]['segments']
                if len(object_segments) > 0: # Only make sense when there are more than one segments
                    filtered_object_segments = filter_object_segment(hand_segments, object_segments)

                    # Becareful duplicated polygons, may cause issue to fillPoly later
                    try:
                        for segment in filtered_object_segments:
                            if segment not in updated_annotations[object_id]:
                                updated_annotations[object_id].append(segment)
                    except KeyError:
                        updated_annotations[object_id] = filtered_object_segments
            
            # Update annotation of a frame
            for object_id, new_segments in updated_annotations.items():
                frames[idx]['annotations'][object_id]['segments'] = new_segments

        relation_map = gen_davis_mask(subseq_name, frames, object_to_color, dataset, out_path, resolution)
        relation_map['left trajectory'] = left_traj
        relation_map['right trajectory'] = right_traj

        # Append new subseq to ImageSets
        # split_path = os.path.join(out_path, 'VISOR_2022', 'ImageSets', '2022', dataset + '.txt')
        # with open(split_path, 'a') as f:
        #     f.write(subseq_name + '\n')

        return color_to_object, relation_map
    
    return None, None


def main(dataset: str, out_path: str, resolution: tuple) -> None:
    """
    Loop through every split annotation file (.json) and group frames by subsequences.
    Process each subsequence to find trajectories and generate DAVIS mask
    """
    # Tracking stats
    num_valid_subseqs = 0 
    
    data_mapping = {}
    relation_mapping = {}
    
    for json_file in tqdm(sorted(glob.glob(os.path.join(SPARSE_ANNO_PATH, dataset) + '/*.json'))):        
        with open(json_file, 'rb') as f:
            data = json.load(f)
        
        # Gathering sub-sequences in a video split
        filtered_anno = filter_annotation(data, True)        
        subseqs = {}
        for frame in filtered_anno['video_annotations']: # Loop for each frame in a split
            subseq_name = frame['image']['subsequence']
            
            # Convert list of annotations to dictionary with id as key for easier query
            new_anno = {}
            for entity in frame['annotations']:
                new_anno[entity['id']] = entity
            
            frame['annotations'] = new_anno

            try:
                subseqs[subseq_name].append(frame)
            except KeyError:
                subseqs[subseq_name] = [frame]
        
        # Process each sub-sequence
        for subseq_name, list_frames in subseqs.items():
            if len(list_frames) < 6:
                # Not enough frames
                continue

            subseq_data_map, subseq_relation_map = process_subseq(subseq_name, list_frames, dataset, out_path, resolution)
            if subseq_data_map:
                num_valid_subseqs += 1
                data_mapping[subseq_name] = subseq_data_map
                relation_mapping[subseq_name] = subseq_relation_map

    # Write data mapping and relation mapping to file
    data_mapping_path = os.path.join(out_path, 'VISOR_DAVIS', dataset, 'data_mapping.json')
    # if os.path.isfile(data_mapping_path):
    #     # If there already exists an annotation json file -> append to it
    #     with open(data_mapping_path, 'r') as f:
    #         read_data_mapping = json.load(f)
    #     read_data_mapping.update(data_mapping)
    
    # (Over)write to file
    with open(data_mapping_path, 'w') as f:
        json.dump(data_mapping, f)
    
    with open(os.path.join(out_path, 'VISOR_DAVIS', dataset, 'hand_relation.json'), 'w') as f:
        json.dump(relation_mapping, f) 
        
    print("==== STATS ====")
    print(f"Number of valid trajectories {num_valid_subseqs}")
    print(f"Number of entities {NUM_ENTITIES}, avg per frame {NUM_ENTITIES/NUM_FRAMES}")
    print(f"Number of frames {NUM_FRAMES}, avg per subseq {NUM_FRAMES/num_valid_subseqs}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parameters for VISOR to DAVIS conversion")
    parser.add_argument('-s', '--set', type=str, help="train or val", required=True, default='train')
    parser.add_argument('-o', '--out', type=str, help="path to the output directory", default='.')
    parser.add_argument('-r', '--resolution', type=str, help="resolution of the output images and masks, in widthxheight format", default='854x480')
    args = parser.parse_args()
    
    out_path = args.out
    source_set = args.set
    resolution = list(map(int, args.resolution.split('x')))
    
    # out_path = '.'
    # source_set = 'train'
    # resolution = (854, 480)
    
    # Making directory for annotations
    # os.makedirs(os.path.join(out_path, 'VISOR_2022', 'Annotations', str(resolution[1]) + 'p'), exist_ok=True)
    # os.makedirs(os.path.join(out_path, 'VISOR_2022', 'ImageSets', '2022'), exist_ok=True)
    # os.makedirs(os.path.join(out_path, 'VISOR_2022', 'JPEGImages', str(resolution[1]) + 'p'), exist_ok=True)
    
    os.makedirs(os.path.join(out_path, 'VISOR_DAVIS', source_set, 'Annotations'), exist_ok=True)
    # os.makedirs(os.path.join(out_path, 'VISOR_DAVIS', 'ImageSets', '2022'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'VISOR_DAVIS', source_set, 'JPEGImages'), exist_ok=True)
    
    main(source_set, out_path, resolution)

