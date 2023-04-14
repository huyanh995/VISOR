"""
Huy Anh Nguyen, CS Stony Brook University
Created on Apr 13, 2023 - 9:37 AM, EDT

Refactor VISOR pre-processing code in OOP style
"""

import os
import json
import glob

import yaml
import cv2
import numpy as np
from tqdm import tqdm
from utils import process_poly, poly_to_bbox

class Frame:
    def __init__(self):
        raise NotImplementedError


class VISORData:
    HAND_IDS = [300, 301] # Left hand, right hand
    GLOVE_IDS = [303, 304] # Left glove, right glove
    LEFT_HAND_IDS = [300, 303]
    RIGHT_HAND_IDS = [301, 304]
    
    GLOVE_VALUES = [['right hand'], ['left hand'], ['left hand', 'right hand']] # Exclude 'inconclusive', None
    INVALID_CONTACTS = ['hand-not-in-contact', 
                        'inconclusive', 
                        'none-of-the-above', 
                        'glove-not-in-contact', 
                    None]
    
    def __init__(self, config, sparse_root, subset = 'val', dense_root = None, dense_img_path = None):
        self.subset = subset
        self.resolution = None
        
        self.sparse_root = sparse_root
        self.sparse_img_root = os.path.join(self.sparse_root, 'rgb_frames', self.subset)
        self.sparse_anno_root = os.path.join(self.sparse_root, 'annotations', self.subset)
        
        # Unpack config
        self.mode = config['coverage_filter']['mode']
        self.max_area = config['coverage_filter']['max_area']
        self.min_area = config['coverage_filter']['min_area']
        self.rm_class_ids = config['static_filter']['categories'] 
        self.rm_names = config['static_filter']['object_names']

        # Dense annotation section
        self.dense_root = dense_root
        if self.dense_root:
            self.dense_anno_root = os.path.join(self.dense_root, self.subset)
        self.dense_img_root = dense_img_path
        
        if dense_root:
            self.fps = config['dense_sample']['fps'] # sample rate for Dense augmentation
        
        self._check_directories()
        self._load_subsequences()

        
    def _check_directories(self):
        """
        Check if the sparse root and (optional) dense_root follow the original VISOR
        """
        if not os.path.exists(self.sparse_root):
            raise FileNotFoundError(f'VISOR Sparse not found in the specified directory')
        
        if not os.path.exists(self.sparse_img_root) or not os.path.exists(self.sparse_anno_root):
            raise FileNotFoundError(f'VISOR Sparse subset {self.subset} not found')
        
        if self.dense_root:
            if not os.path.exists(self.dense_root):
                raise FileNotFoundError(f'VISOR Dense not found in the specified directory')
            if not os.path.exists(self.dense_mask_root):
                raise FileNotFoundError(f'VISOR Dense Interpolation subset {self.subset} not found')
            if not os.path.exists(self.dense_img_path):
                raise FileExistsError(f'VISOR Dense Frames not found')
            
        
    def _load_subsequences(self):
        """
        This function should process **frames** only
        """
        self.subsequences = dict()
        
        for json_file in tqdm(sorted(glob.glob(os.path.join(self.sparse_anno_root, '*.json')))):
            with open(json_file, 'r') as f:
                data = json.load(f)
        
            for frame in data['video_annotations']: # Loop over frames in a sequence
                subseq_name = frame['image']['subsequence']
                frame['name'] = frame['image']['name'].replace('.jpg', '')
                
                # Update image_path and mask_path
                frame['image_path'] = os.path.join(self.sparse_img_root, 
                                                        frame['image']['video'].split('_')[0], 
                                                        frame['image']['image_path'])

                if not self.resolution: # get resolution 
                    self.resolution = cv2.imread(frame['image_path']).shape[:-1]
                    
                del frame['image']
                
                # # Extract hand and object relation in each frame
                # updated_annotations = {}
                # for entity in frame['annotations']:
                #     # Extract hands and objects in hand id
                #     entity_id = entity['id']
                #     if entity['class_id'] in VISORData.LEFT_HAND_IDS + VISORData.RIGHT_HAND_IDS:
                #         side = 'left' if entity['class_id'] in VISORData.LEFT_HAND_IDS else 'right'
                #         entity.setdefault(f'{side} hand', []).append(entity_id)

                #         # entity must have 'in_contact_object' field, check for its value
                #         if entity['in_contact_object'] not in VISORData.INVALID_CONTACTS:
                #             # TODO: check if in_contact_object is glove
                #             entity.setdefault('relations', []).append((entity_id, entity['in_contact_object']))
                #             entity.setdefault(f'{side} object', []).append(entity['in_contact_object'])
                        
                #     else:
                #         if not self._filter_object(entity):
                #             continue # skip the not passed objects
                        
                #     updated_annotations[entity_id] = entity
                
                # frame['annotations'] = updated_annotations # Change from list to dict
                
                self.subsequences.setdefault(subseq_name, []).append(frame)
                
        
    def _filter_object(self, entity):
        """
        Filter object based on its size or black list
        Input: entity in a frame
        """
        # Filter based on black list
        if entity['class_id'] in self.rm_class_ids \
            or entity['name'] in self.rm_names:
            return False
            
        # Filter based on coverage area
        mask = np.zeros(self.resolution)
        
        if self.mode == 'mask':
            polygons = process_poly(entity['segments'])
            cv2.fillPoly(mask, polygons, (1, 1, 1))
            coverage = np.sum(mask) / np.prod(self.resolution)
            
        elif self.mode == 'bbox':
            bboxes = poly_to_bbox(entity['segments'])
            for bbox in bboxes:
                bbox = list(map(round, bbox)) # convert float to int
                cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 1, -1) # -1 mean fill entirely rectangle
            coverage = np.sum(mask) / np.prod(self.resolution)
        
        if self.min_area > coverage or self.max_area < coverage:
            return False
        return True


    def _process_frame(self, frame):
        """
        Filter entities based on rules:
        - Always keep hands and gloves (not the category 60 glove)
        - Only keep object-in-hand that:
            - Not in black-list categories and items
            - Have reasonable size based on its mask or bbox area
            Both are in config file
            
        TODO: 
            []
            
        """
        # Extract hand and object relation in each frame
        
        entities = {entity['id']: entity for entity in frame['annotations']} # for quick query
        updated_annotations = {}
        for entity_id, entity in entities.items():
            # Extract hands and objects in hand id
            if entity['class_id'] in VISORData.LEFT_HAND_IDS + VISORData.RIGHT_HAND_IDS:
                side = 'left' if entity['class_id'] in VISORData.LEFT_HAND_IDS else 'right'
                frame.setdefault(f'{side} hand', []).append(entity_id)

                # entity must have 'in_contact_object' field, check for its value
                if (object_id := entity['in_contact_object']) not in VISORData.INVALID_CONTACTS:
                    # Check if in_contact_object is glove
                    if entities[object_id]['class_id'] not in VISORData.GLOVE_IDS:
                        frame.setdefault('relations', []).append((entity_id, object_id))
                        frame.setdefault(f'{side} object', []).append(object_id)
                    else:
                        # DEBUG region
                        # Check glove is on designed hand
                        # And have 'in_contact_object'
                        if (on_which_hand := entities[object_id]['on_which_hand']) in VISORData.GLOVE_VALUES:
                            if f'{side} hand' not in entities[object_id]['on_which_hand']:
                                print(f'Wrong hand: {frame["name"]}')
                                VISORData.draw_frame(frame)
                            if len(entities[object_id]['on_which_hand']) > 1:
                                print(f'On both hands {frame["name"]}')
                            
                        assert 'in_contact_object' in entities[object_id], 'gloves lack attribute'
        
                
            else:
                if not self._filter_object(entity):
                    continue # skip the not passed objects
                
            updated_annotations[entity_id] = entity
        
        frame['annotations'] = updated_annotations
        
        return frame
        
    def _sample_frames(self, subseq):
        """
        Rough ideas:
            [] Only preprocessing dense annotation, check if associated frames exists
        """
        
        raise NotImplementedError
    
    
    def get_subsequence(self, augment=False):
        if augment:
            assert self.dense_root and self.dense_img_root, 'Dense annotations and dense frames not found'
            
        for name, frames in self.subsequences.items():
            trajectory = {'left': [], 'right': []}
            # for idx in range(len(frames)):
            #     frames[idx] = self._process_frame(frames[idx])

            for idx, frame in enumerate(frames):
                # Process each frame: dict, frames: list
 
                frames[idx] = self._process_frame(frame)
                # Extract object-in-hand
                left_objects = frame.get('left object', [])
                right_objects = frame.get('right object', [])
                
                if len(left_objects) > 1 or len(right_objects) > 1:
                    print(f'Found {name}')
                    
                # trajectory['left'].append(frame.get('left hand'))
                # trajectory['right'].append(frame.get('right hand'))
                
            subseq = {'name': name, 
                    'frames': frames,
                    'trajectories': trajectory,
                    }

            # return subseq
    
    def get_subsequence_names(self):
        return list(self.subsequences.keys())
    
    def __getitem__(self, name):
        """
        Get raw annotation based on subseq name
        """
        frames = self.subsequences[name]
        for idx in range(len(frames)):
            frames[idx] = self._process_frame(frames[idx])
            
        return frames
    
    # For debugging
    @staticmethod
    def draw_frame(frame):
        # Draw a mask and overlay its on images
        colors = [
                    (128, 0, 0),    # maroon
                    (255, 0, 0),    # red
                    (0, 255, 0),    # lime
                    (255, 255, 0),  # yellow
                    (0, 0, 255),    # blue
                    (255, 0, 255),  # fuchsia
                    (128, 0, 128),  # purple
                    (0, 255, 255),  # aqua
                    (0, 128, 128),  # teal
                ]
        file_name = frame['name'] + '.jpg'
        img = cv2.imread(frame['image_path'])
        h, w = img.shape[:-1]
        mask = np.zeros_like(img)
        
        entities = {entity['id']: entity for entity in frame['annotations']}
        for idx, (entity_id, entity) in enumerate(entities.items()):
            polygons = process_poly(entity['segments'])
            name = entity['name']
            color = colors[idx % len(colors)]
            cv2.fillPoly(mask, polygons, color=color)
            
            # Display text on image 
            text = entity['name']
            if 'on_which_hand' in entity and entity['on_which_hand']:
                text = text + ' On:' + ' '.join(entity['on_which_hand'])
            if 'in_contact_object' in entity and entity['in_contact_object'] not in VISORData.INVALID_CONTACTS:
                text = text + ' In:' + entities[entity['in_contact_object']]['name']
            
            # Get text location
            bboxes = poly_to_bbox(entity['segments'])
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.1
            font_thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            x1, y1, x2, y2 = bboxes[0]
            x_text = int(min((x1 + x2) / 2, w - text_w - 10)) # center of witdth
            y_text = int(max(y1 - 10, 40))
            cv2.putText(img, text, (x_text, y_text), cv2.FONT_HERSHEY_DUPLEX, 1.1, color=color, thickness=2)
            
        # Overlay mask on image
        overlay = cv2.addWeighted(mask, 0.35, img, 1, 0)
        
        # Write image
        cv2.imwrite(file_name, overlay)


if __name__ == '__main__':
    SPARSE_ROOT = '../GroundTruth-SparseAnnotations'
    DENSE_ROOT = None
    DENSE_IMG_ROOT = None
    
    SUBSET = 'train'
    
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Check config validity
    assert config['coverage_filter']['mode'] in ['bbox', 'mask'], 'Coverage mode must be either bbox or mask'
    assert 0 <= config['coverage_filter']['max_area'] <= 1, 'Max area coverage must be in range [0, 1]'
    assert 0 <= config['coverage_filter']['min_area'] <= 1, 'Min area coverage must be in range [0, 1]'

    data = VISORData(config, SPARSE_ROOT, SUBSET, DENSE_ROOT, DENSE_IMG_ROOT)
    data.get_subsequence()
    print("DEBUG")

