"""
Huy Anh Nguyen, CS Stony Brook University
Created on Mar 9, 2023 - 8:13 PM, EDT

Filtering the annotation json file. Only keep the hand annotation and its hand-held objects
JSON file format: each file is for one sequence (or clip)
"""

import json
import os
import glob
from collections import Counter
import argparse

import yaml
import numpy as np
import cv2
from tqdm import tqdm # TODO: Think about remove it later for simplicity?

from utils import process_poly, poly_to_bbox

SPARSE_ANNO_PATH = '../GroundTruth-SparseAnnotations/annotations'
HAND_IDS = [300, 301] # Left hand, right hand
GLOVE_IDS = [303, 304, 60] # Left glove, right glove
GLOVE_VALUES = [['right hand'], ['left hand'], ['left hand', 'right hand']] # Exclude 'inconclusive', None
INVALID_CONTACTS = ['hand-not-in-contact', 
                    'inconclusive', 
                    'none-of-the-above', 
                    'glove-not-in-contact', 
                    None]

def load_config(filename = 'config.yml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
        
    # Check config validity
    assert config['coverage_filter']['mode'] in ['bbox', 'mask'], 'Coverage mode must be either bbox or mask'
    assert 0 <= config['coverage_filter']['max_area'] <= 1, 'Max area coverage must be in range [0, 1]'
    assert 0 <= config['coverage_filter']['min_area'] <= 1, 'Min area coverage must be in range [0, 1]'
    
    return config   

def filter_annotation(data: dict, config: dict) -> dict:
    """
    Process each video cli (sequence) annotation at FRAMEs level. 
    - Filter out all non hand-held objects.
    - Ensure that a hand-held object does not cover large than 25% area of a frame...
    - ...can be calculated through union of masks or bounding boxes.
    """
    # Stats variable
    stats = {}
    stats['num_frames'] = len(data['video_annotations'])
    # stats['num_objects'] = 0 # Consider remove it since objects will cover this stat
    stats['objects'] = {} # dictionary of objects and its coverage to update
    stats['hands'] = []
    
    # Unpack config
    mode = config['coverage_filter']['mode']
    max_area = config['coverage_filter']['max_area'] 
    min_area = config['coverage_filter']['min_area']
    
    filtered_categories = config['static_filter']['categories'] 
    filtered_objects = config['static_filter']['object_names']
    
    # Process data (list of frames)
    
    res = {'info': data['info'], 'video_annotations': []}

    for img_anno in data['video_annotations']: # Loop over image/frame in a sequence
        
        ##### PART 1: ONLY SELECT OBJECT ON HANDS #####
        keep_hands_ids = set()
        keep_object_ids = set()
        relation = []
        
        # Store left and right hand-held object
        left_object = None
        right_object = None
        
        # Store left and right hand/glove id to merge them later
        left_hands = []
        right_hands = []
        
        # Store id of hands and gloves to process later
        left_hand_id = None
        right_hand_id = None
        gloves = []
        
        # Convert list entities to dict for easier query
        entities = {}
        for entity in img_anno['annotations']:
            if entity['class_id'] == HAND_IDS[0]: 
                left_hand_id = entity['id'] # left hand
                left_hands.append(left_hand_id)
                
            elif entity['class_id'] == HAND_IDS[1]: 
                right_hand_id = entity['id'] # right hand
                right_hands.append(right_hand_id)
                
            elif entity['class_id'] in GLOVE_IDS \
                    and 'on_which_hand' in entity \
                    and entity['on_which_hand'] in GLOVE_VALUES: # Add gloves which are on hand(s) to process later
                gloves.append((entity['id'], entity['on_which_hand']))
                
            entities[entity['id']] = entity
                
        if gloves: # there is glove in the frame
            for (glove_id, on_which_hand) in gloves:
                # on_which_hand is a list of which hand the glove is on. Can be non, one hand, or both hands
                
                # keep_ids.add(glove_id) # DEBUG: Always keep hands and gloves 
                keep_hands_ids.add(glove_id) # Always keep gloves
                glove_object = entities[glove_id]
                # Only matter if glove is in contact with some objects
                if 'in_contact_object' in glove_object and glove_object['in_contact_object'] not in INVALID_CONTACTS:
                    object_id = glove_object['in_contact_object']
                else:
                    continue
                
                # General idea, assuming left hand and glove is in contact with an object
                # - If left hand appears in the frame and glove on_which_hand is left hand...
                # ...than left hand in_contact_with should be the glove id
                # - If left hand is not in the frame so just go ahead
                
                is_left_valid = (left_hand_id and entities[left_hand_id]['in_contact_object'] == glove_id) \
                                or (not left_hand_id and 'left hand' in on_which_hand)

                is_right_valid = (right_hand_id and entities[right_hand_id]['in_contact_object'] == glove_id) \
                                or (not right_hand_id and 'right hand' in on_which_hand)
                                
                if len(on_which_hand) > 1: # glove on both hands
                    # TODO: How to merge hand and glove if glove is on both hands?
                    # Luckily there isn't this case in EPIC VISOR dataset...
                    # ...but may still consider later
                    
                    # left_hands.append(glove_id)
                    # right_hands.append(glove_id)
                    # Add some debug here
                    if is_left_valid and is_right_valid:
                        print("DEBUG")
                    
                    if is_left_valid:
                        relation.append((glove_id, object_id))
                        # keep_ids.add(object_id) # DEBUG 
                        keep_object_ids.add(object_id) 
                        left_object = object_id

                    if is_right_valid:
                        relation.append((glove_id, object_id))
                        # keep_ids.add(object_id) # DEBUG
                        keep_object_ids.add(object_id)
                        right_object = object_id

                elif on_which_hand[0] == 'left hand':
                    left_hands.append(glove_id)
                    if is_left_valid:
                        relation.append((glove_id, object_id))
                        # keep_ids.add(object_id)   
                        keep_object_ids.add(object_id)
                        left_object = object_id
                    
                elif on_which_hand[0] == 'right hand':
                    right_hands.append(glove_id)
                    if is_right_valid:
                        relation.append((glove_id, object_id))
                        # keep_ids.add(object_id)
                        keep_object_ids.add(object_id)
                        right_object = object_id
                    else:
                        if img_anno['image']['name'] == 'P06_13_frame_0000002608.jpg':
                            # This image has wrong annotation. Right glove holding pot
                            relation.append((glove_id, object_id))
                            # keep_ids.add(object_id)
                            keep_object_ids.add(object_id)
                            right_object = object_id

        if left_hand_id:
            keep_hands_ids.add(left_hand_id)
            if 'in_contact_object' in entities[left_hand_id] \
                        and entities[left_hand_id]['in_contact_object'] not in INVALID_CONTACTS:
                object_id = entities[left_hand_id]['in_contact_object']
                if entities[object_id]['class_id'] not in GLOVE_IDS:
                    keep_object_ids.add(object_id)
                    relation.append((left_hand_id, object_id))
                    left_object = object_id
                
        if right_hand_id:
            keep_hands_ids.add(right_hand_id)
            if 'in_contact_object' in entities[right_hand_id] \
                        and entities[right_hand_id]['in_contact_object'] not in INVALID_CONTACTS:
                object_id = entities[right_hand_id]['in_contact_object']
                if entities[object_id]['class_id'] not in GLOVE_IDS:
                    if img_anno['image']['name'] == 'P06_13_frame_0000002608.jpg' and right_object:
                        # This image has wrong annotation. Skip it!
                        continue
                    keep_object_ids.add(object_id)
                    relation.append((right_hand_id, object_id))
                    right_object = object_id

        # Remove object not in kept set
        new_anno = {'image': img_anno['image'], 'annotations': [], 'relations': [], 'left object': None, 'right object': None, 'left hand': left_hands, 'right hand': right_hands} # New annotations for a frame
        relation = set(relation)

        for item in img_anno['annotations']:
            if item['id'] in keep_hands_ids:
                new_anno['annotations'].append(item)
                stats['hands'].append(item['name'])

            if item['id'] in keep_object_ids:
                # Add checking for MAX_AREA to be a valid object -> This part may take a lot of additional time
                mask = np.zeros((1080, 1920), dtype=np.uint8)

                if mode == 'mask':
                    # Using mask coverage
                    polygons = process_poly(item['segments'])
                    cv2.fillPoly(mask, polygons, (1, 1, 1)) # Should be a binary mask of that object
                    area_coverage = np.sum(mask) / (1080 * 1920)   

                elif mode == 'bbox':
                    # Using union of bboxes
                    bboxes = poly_to_bbox(item['segments'])
                    for bbox in bboxes:
                        bbox = list(map(round, bbox)) # convert float to int
                        cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 1, -1) # -1 mean fill entirely rectangle
                    area_coverage = np.sum(mask) / (1080 * 1920)
                
                # is_valid_object = (remove_static_obj and item['class_id'] not in STATIC_CLASS \
                #                     and item['name'] not in STATIC_NAMES ) \
                #                     or not remove_static_obj
                
                is_valid_object = item['class_id'] not in filtered_categories \
                                    and item['name'] not in filtered_objects
                                    
                if is_valid_object and min_area <= area_coverage <= max_area:
                    # Add to stats var
                    if item['name'] in stats['objects']:
                        stats['objects'][item['name']].append(area_coverage)
                    else:
                        stats['objects'][item['name']] = [area_coverage]
                    
                    # Add valid object annotation to new dictionary
                    new_anno['annotations'].append(item)
                    # Update left and right annotation
                    if item['id'] == left_object:
                        new_anno['left object'] = item['name']

                    if item['id'] == right_object:
                        new_anno['right object'] = item['name']
                    
                    # Update relation annotation
                    for hand_id, object_id in relation:
                        if object_id == item['id']: # item found in relation
                            new_anno['relations'].append((hand_id, object_id))
        
        res['video_annotations'].append(new_anno)
        
    return res, stats

            
def merge_stats(ls_stats: list) -> dict:
    assert len(ls_stats) > 1, 'Cannot be an empty list'
    # keys should be objects, num_frames, hands
    res = {'num_frames': 0, 'num_objects': 0, 'objects': {}, 'hands': []}
    for stats in ls_stats: # Accummulate the stats
        res['num_frames'] += stats['num_frames']
        res['hands'] += stats['hands']
        # objects is a dict of coverage
        for key in stats['objects']:
            res['num_objects'] += len(stats['objects'][key])
            if key in res['objects']:
                res['objects'][key] += stats['objects'][key]
            else:
                res['objects'][key] = stats['objects'][key]
    
    # Post processing
    res['hands'] = dict(Counter(res['hands']))
    tmp = {k: f'{len(v)} {round(float(sum(v) / len(v)), 5)}' for k, v in res['objects'].items()}
    res['objects'] = tmp
    
    return res
        
        
def main(dataset: str, out_path: str):
    # Load config
    config = load_config()
    ls_stats = []
    out_anno_dir = os.path.join(out_path, 'filtered_annotations')
    os.makedirs(out_anno_dir, exist_ok=True)

    for json_file in tqdm(sorted(glob.glob(os.path.join(SPARSE_ANNO_PATH, dataset) + '/*.json'))):
        with open(json_file, 'rb') as f:
            data = json.load(f)
        
        filtered_anno, stats = filter_annotation(data, config)
        ls_stats.append(stats)
        with open(os.path.join(out_anno_dir, json_file.split('/')[-1]), 'w') as f:
            json.dump(filtered_anno, f)

    stats = (merge_stats(ls_stats))
    
    if config['stats']:
        # dump stats into a yaml file
        with open(os.path.join(out_path, f'{dataset}_stats.yml'), 'w') as f:
            yaml.dump(stats, f)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parameters for filter VISOR annotations")
    parser.add_argument('-s', '--set', type=str, help="train or val, or both", default='val')
    parser.add_argument('-o', '--out', type=str, help="path to the output directory", default='.')
    args = parser.parse_args()
    
    assert args.set in ['train', 'val', 'both'], '--set must be one of train/val/both'
    if args.set == 'train':
        print(">>> TRAIN SET")
        main('train', args.out)
    elif args.set == 'val':
        print(">>> VAL SET")
        main('val', args.out)
    else:
        print(">>> BOTH SETS")
        main('train', args.out)
        main('val', args.out)
