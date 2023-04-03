"""
Huy Anh Nguyen, CS Stony Brook University
Created on Mar 13, 2023 - 1:41 AM, EDT

Generate trajectories from VISOR annotation
TODO:
    [] Add remove black frames feature
"""

import json
import os
import glob
import argparse
from collections import Counter
import yaml
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image

from tqdm import tqdm
from filter_hand_object import filter_annotation, load_config, HAND_IDS, GLOVE_IDS
from utils import process_poly

SPARSE_ANNO_PATH = '../GroundTruth-SparseAnnotations/annotations'
SPARSE_IMG_PATH = '../GroundTruth-SparseAnnotations/rgb_frames'
HANDS_MAPPING = {'left hand': 1,
                'right hand': 2,
                'left glove': 3,
                'right glove': 4,
                'glove': 5,
                }


def is_trajectory(traj: list, window: int = 2) -> bool:
    res = False
    for i in range(window, len(traj) + 1):
        tmp_set = set(traj[i - window : i])
        if len(tmp_set) == 1 and tmp_set != set([None]):
            res = True
    return res


def dilate(mask, iterations = 5):
    mask = ndimage.binary_dilation(mask, iterations = iterations).astype(mask.dtype)
    return mask


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
    

def gen_davis_mask(subseq_name: str, frames:list, object_to_color: dict, dataset: str, out_path: str, resolution: tuple, hand: bool):
    stats = {'num_frames': len(frames),
             'num_objects': 0,
             'hands': [],
             'objects': {},
             'categories': {}
            }
    
    # Not count hands by default
    h, w = resolution
    anno_path = os.path.join(out_path, 'VISOR_2022', 'Annotations', str(resolution[1]) + 'p', subseq_name)
    image_path = os.path.join(out_path, 'VISOR_2022', 'JPEGImages', str(resolution[1]) + 'p', subseq_name)
    
    # Create folders for new subsequence
    os.makedirs(anno_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)

    relation_map = {}
    
    for idx, frame in enumerate(frames):
        stats['num_objects'] += len(frame['annotations'])
        
        frame_name = f'{idx:05}'
        
        # Generate relation map  
        relation_map[frame_name + '.jpg'] = {}
        if hand:
            relation_map[frame_name + '.jpg']['relations'] = []
            for hand_id, object_id in frame['relations']:
                if hand_id in frame['left hand']:
                    hand_name = 'left hand'
                elif hand_id in frame['right hand']:
                    hand_name = 'right hand'
                relation_map[frame_name + '.jpg']['relations'].append((object_to_color[hand_name], \
                                                                object_to_color[frame['annotations'][object_id]['name']]))

        else:
            relation_map[frame_name + '.jpg']['relations'] = [(HANDS_MAPPING[frame['annotations'][k]['name']], \
                                                object_to_color[frame['annotations'][v]['name']] + 5) \
                                                for k, v in frame['relations']]

        # Generate mask
        img_path = os.path.join(SPARSE_IMG_PATH, dataset, frame['image']['video'].split('_')[0], frame['image']['image_path'])
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)
        
        mask = np.zeros(img.shape[:-1], dtype = np.uint8)
        stats_objects = []
        # Get the polygons of hand-held objects from relation annotation
        for hand_id, object_id in frame['relations']:
            object_name = frame['annotations'][object_id]['name']
            object_cat = frame['annotations'][object_id]['class_id']
            
            object_polygons = process_poly(frame['annotations'][object_id]['segments'])
            # Create object mask using collected polygons
            object_color = object_to_color[object_name]
            cv2.fillPoly(mask, object_polygons, (object_color, object_color, object_color))
            
            # Calculating area coverage to add to stats. Notice that the resolution is 1920x1080
            stats_objects.append((object_name, object_cat, object_color))
            
            # Create hand mask if requested
            if hand:
                if hand_id in frame['left hand']:
                    if len(frame['left hand']) > 1:
                        # Need to merge. Each segment is a list of region already
                        # Need to use update list instead of appending
                        hand_polygons = []
                        for tmp_id in frame['left hand']:
                            hand_polygons += process_poly(frame['annotations'][tmp_id]['segments'])
                    else:
                        hand_polygons = process_poly(frame['annotations'][hand_id]['segments'])
                        
                    hand_color = object_to_color['left hand']
                    cv2.fillPoly(mask, hand_polygons, (hand_color, hand_color, hand_color))
                    stats_objects.append(('left hand', 300, hand_color))
                    
                if hand_id in frame['right hand']:
                    if len(frame['right hand']) > 1:
                        hand_polygons = []
                        for tmp_id in frame['right hand']:
                            hand_polygons += process_poly(frame['annotations'][tmp_id]['segments'])
                    else:
                        hand_polygons = process_poly(frame['annotations'][hand_id]['segments'])
                        
                    hand_color = object_to_color['right hand']
                    cv2.fillPoly(mask, hand_polygons, (hand_color, hand_color, hand_color))
                    stats_objects.append(('right hand', 301, hand_color))
        
            
        # Resize mask to match with img
        resized_mask = cv2.resize(mask, resolution, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        # Calculate area coverage and add to stats
        for (name, cat, color) in stats_objects:
            area_coverage = np.sum(np.where(resized_mask == color, 1, 0)) / (h * w)
            if name in stats['objects']:
                stats['objects'][name].append(area_coverage)
            else:
                stats['objects'][name] = [area_coverage]
            
            if cat in stats['categories']:
                stats['categories'][cat].append(name)
            else:
                stats['categories'][cat] = [name]
        # Write annotation mask into Annotations and original image to JPEGImages
        imwrite_davis(os.path.join(anno_path, frame_name + '.png'), resized_mask)
        cv2.imwrite(os.path.join(image_path, frame_name + '.jpg'), resized_img)

    return relation_map, stats


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


def process_subseq(subseq_name: str, list_frames: list, dataset: str, out_path: str, resolution: tuple, hand: bool) -> None:
    h, w = resolution
    # Get list of frames from a subsequence
    frames = sorted(list_frames, key=lambda k: k['image']['image_path'])
    left_traj = [frame['left object'] for frame in frames]
    right_traj = [frame['right object'] for frame in frames]

    # Check if any trajectory is valid
    if is_trajectory(left_traj) or is_trajectory(right_traj):
        # Get set of objects in a valid subsequence
        # Reserve 1 for left hand, 2 for right hand
        set_objects = set(left_traj + right_traj)
        set_objects.discard(None)
        if hand:
            # Check if left or right hand in set
            tmp_color_to_object = {}
            for frame in frames:
                if frame['left hand']:
                    tmp_color_to_object[1] = 'left hand'
                if frame['right hand']:
                    tmp_color_to_object[2] = 'right hand'
            # TODO: Experimental: Make an adjustment in case only left hand or right hand in the subsequence
            offset = 3
            if len(tmp_color_to_object.keys()) == 1:
                if 'right hand' in tmp_color_to_object.values():
                    # Only right hand with color 2 in subseq.
                    # Move right hand to 1
                    del tmp_color_to_object[2]
                    tmp_color_to_object[1] = 'right hand'
                offset = 2
            
            # Sorted the dictionary for nicer file output
            tmp_keys = sorted(tmp_color_to_object.keys())
            color_to_object = {k: tmp_color_to_object[k] for k in tmp_keys}
            
            for k, obj in enumerate(sorted(set_objects)):
                color_to_object[k + offset] = obj

        else:
            color_to_object = {k + 1: obj for k, obj in enumerate(sorted(set_objects))} # Try to get some order
        object_to_color = {v: k for k, v in color_to_object.items()}
        
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
            
            # Update objects annotation of a frame
            for object_id, new_segments in updated_annotations.items():
                frames[idx]['annotations'][object_id]['segments'] = new_segments
                
        relation_map, subseq_stats = gen_davis_mask(subseq_name, frames, object_to_color, dataset, out_path, resolution, hand)
        relation_map['left trajectory'] = left_traj
        relation_map['right trajectory'] = right_traj

        # Append new subseq to ImageSets
        split_path = os.path.join(out_path, 'VISOR_2022', 'ImageSets', '2022', dataset + '.txt')
        with open(split_path, 'a') as f:
            f.write(subseq_name + '\n')

        return color_to_object, relation_map, subseq_stats
    
    return None, None, None


def main(dataset: str, out_path: str, resolution: tuple, hand: bool) -> None:
    """
    Loop through every split annotation file (.json) and group frames by subsequences.
    Process each subsequence to find trajectories and generate DAVIS mask
    """
    # Loading config
    config = load_config()
    
    data_mapping = {}
    relation_mapping = {}
    
    stats = {'num_valid_subseqs': 0,
             'num_frames': 0,
             'num_objects': 0,
             'hands': [],
             'objects': {},
             'categories': {}
            }

    for json_file in tqdm(sorted(glob.glob(os.path.join(SPARSE_ANNO_PATH, dataset) + '/*.json'))):     
        with open(json_file, 'rb') as f:
            data = json.load(f)
        
        # Gathering sub-sequences in a video split
        filtered_anno, _ = filter_annotation(data, config)        
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

            subseq_data_map, subseq_relation_map, subseq_stats = process_subseq(subseq_name, list_frames, dataset, out_path, resolution, hand)
            if subseq_data_map: # found a valid trajectory
                # Update stats
                stats['num_valid_subseqs'] += 1
                stats['num_frames'] += subseq_stats['num_frames']
                stats['num_objects'] += subseq_stats['num_objects']
                stats['hands'] += subseq_stats['hands']
                
                # Update objects and categories dict
                for key in subseq_stats['objects']:
                    if key in stats['objects']: stats['objects'][key] += subseq_stats['objects'][key]
                    else: stats['objects'][key] = subseq_stats['objects'][key]
                        
                for key in subseq_stats['categories']:
                    if key in stats['categories']: stats['categories'][key] += subseq_stats['categories'][key]
                    else: stats['categories'][key] = subseq_stats['categories'][key]
                                        
                data_mapping[subseq_name] = subseq_data_map
                relation_mapping[subseq_name] = subseq_relation_map

    # Write data mapping and relation mapping to file
    data_mapping_path = os.path.join(out_path, 'VISOR_2022', f'data_mapping.json')
    if os.path.isfile(data_mapping_path):
        # If there already exists an annotation json file -> append to it
        with open(data_mapping_path, 'r') as f:
            read_data_mapping = json.load(f)
        data_mapping.update(read_data_mapping)
    
    with open(data_mapping_path, 'w') as f:
        json.dump(data_mapping, f)
    
    hand_relation_path = os.path.join(out_path, 'VISOR_2022', f'hand_relation.json')
    if os.path.isfile(hand_relation_path):
        with open(hand_relation_path, 'r') as f:
            read_hand_realtion = json.load(f)
        relation_mapping.update(read_hand_realtion)
        
    with open(hand_relation_path, 'w') as f:
        json.dump(relation_mapping, f) 
        
    # Output stats file
    if config['stats']:
        stats['hands'] = dict(Counter(stats['hands']))
        tmp = {k: round(float(sum(v) / len(v)), 2) for k, v in stats['objects'].items()}
        stats['objects'] = tmp
        tmp = {k: dict(Counter(v)) for k, v in stats['categories'].items()}
        stats['categories'] = tmp
        
        # Write to yml file
        with open(os.path.join(out_path, f'{dataset}_stats.yml'), 'w') as f:
            yaml.dump(stats, f, sort_keys=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parameters for VISOR to DAVIS conversion")
    parser.add_argument('-s', '--set', type=str, help="train or val", default='val')
    parser.add_argument('-o', '--out', type=str, help="path to the output directory", default='.')
    parser.add_argument('-r', '--resolution', type=str, help="resolution of the output images and masks, in widthxheight format", default='854x480')
    parser.add_argument('-hand', action='store_true', help="add hand to the output file, note that hand and glove on that hand will be merged", default=False)
    args = parser.parse_args()
    
    out_path = args.out
    source_set = args.set
    resolution = list(map(int, args.resolution.split('x')))
    hand = args.hand
    
    # Making directory for annotations
    os.makedirs(os.path.join(out_path, 'VISOR_2022', 'Annotations', str(resolution[1]) + 'p'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'VISOR_2022', 'ImageSets', '2022'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'VISOR_2022', 'JPEGImages', str(resolution[1]) + 'p'), exist_ok=True)
    
    main(source_set, out_path, resolution, hand)

