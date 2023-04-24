"""
Huy Anh Nguyen, CS Stony Brook University
Created on Apr 13, 2023 - 9:37 AM, EDT

Refactor VISOR pre-processing code in OOP style
Some observations:
* all hands (id 300, 301) have 'in_contact_object' but NOT 'on_which_hand'
* all gloves (id 303, 304) have 'in_contact_object' and 'on_which_hand'
* SOME of glove (id 60) have 'in_contact_object' and 'on_which_hand'
* id 60 has many instance names, not just 'glove'
"""

import os
import json
import glob
import logging
import time

import yaml
import cv2
import numpy as np
from tqdm import tqdm
from utils import process_poly, poly_to_bbox, calculate_segment_area

logging.basicConfig(level=logging.ERROR)

class VISORData:
    HAND_IDS: list = [300, 301] # Left hand, right hand
    GLOVE_IDS: list = [303, 304] # Left glove, right glove
    LEFT_HAND_IDS: list = [300, 303]
    RIGHT_HAND_IDS: list = [301, 304]
    
    GLOVE_VALUES: list = [['right hand'], ['left hand'], ['left hand', 'right hand']] # Exclude 'inconclusive', None
    INVALID_CONTACTS: list = ['hand-not-in-contact', 
                            'inconclusive', 
                            'none-of-the-above', 
                            'glove-not-in-contact', 
                            None]
    
    def __init__(self, 
                 config: dict,
                 sparse_root: str,
                 subset: str = 'val', 
                 dense_root: str = None, 
                 dense_img_path: str = None):
        
        self.name = 'VISOR'
        self.year = '2022' # To pair with DAVIS evaluation code
        
        self.subset = subset
        self.resolution = None
        
        self.sparse_root = sparse_root
        self.sparse_img_root = os.path.join(self.sparse_root, 'rgb_frames', self.subset)
        self.sparse_anno_root = os.path.join(self.sparse_root, 'annotations', self.subset)
        
        # Unpack config
        self.mode = config['coverage_filter']['mode']
        self.max_area = config['coverage_filter']['max_area']
        self.min_area = config['coverage_filter']['min_area']
        self.bl_class_ids = config['static_filter']['categories']   # black-listed categories
        self.bl_names = config['static_filter']['object_names']     # black-listed names
        
        # Dense annotation section
        self.dense_root = dense_root
        if self.dense_root:
            # self.dense_anno_root = os.path.join(self.dense_root, self.subset)
            self.dense_json_files = sorted(glob.glob(os.path.join(self.dense_root, self.subset, '*', '*.json')))
            self.dense_anno = (None, {}) # current dense annotation, because it's so huge, can not load all at once
            self.desired_fps = config['dense_sample']['fps'] # sample rate for Dense augmentation
            self.dense_img_root = dense_img_path
            
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

            # if not os.path.exists(self.dense_img_path):
            #     raise FileExistsError(f'VISOR Dense Frames not found')
        
        
    def _load_subsequences(self) -> None:
        """
        Load sparse annotation .json file and group frames into self.subsequence
        """
        self.subsequences = dict()
        
        # For stats of original dataset
        n_objects = 0
        n_frames = 0
        
        for json_file in tqdm(sorted(glob.glob(os.path.join(self.sparse_anno_root, '*.json')))):
            with open(json_file, 'r') as f:
                data = json.load(f)

            for frame in data['video_annotations']: # Loop over frames in a sequence
                # Manually correction for annotations
                if 'P04_121_frame_0000075601' in frame['image']['name']:
                    for entity in frame['annotations']:
                        if entity['id'] == '6e234d3c52ecafb034c199642f6373eb':
                            entity['on_which_hand'][-1] = 'right hand' # typo: rigth 

                #####################################
                subseq_name = frame['image']['subsequence']
                frame['name'] = frame['image']['name'].replace('.jpg', '')
                frame['subsequence'] = frame['image']['subsequence'] # FIXME: for debug only, remove later if needed
                
                # Update image_path to absolute path
                frame['image_path'] = os.path.join(self.sparse_img_root, 
                                                        frame['image']['video'].split('_')[0], 
                                                        frame['image']['image_path'])

                if not self.resolution: # get resolution
                    self.resolution = cv2.imread(frame['image_path']).shape[:-1]
                    
                del frame['image']
                n_objects += len(frame['annotations'])
    
                self.subsequences.setdefault(subseq_name, []).append(frame)
            n_frames += len(data['video_annotations'])
                
        # Display stats after loading dataset annotations
        print("-" * 20)
        print("Stats for original {} set".format(self.subset))
        print("Total subsequences: {}".format(len(self.subsequences)))
        print("Total frames: {}".format(n_frames))
        print("Total objects: {}".format(n_objects))
        print("-" * 20)
    
    def _filter_object(self, entity: dict) -> bool:
        """
        Filter object based on its size or black list
        Input: entity in a frame
        """
        # Filter based on black list
        if entity['class_id'] in self.bl_class_ids \
            or entity['name'] in self.bl_names:
            return False
            
        # Filter based on coverage area
        # coverage = calculate_segment_area(entity['segments'], self.resolution, self.mode) / np.prod(self.resolution)
        coverage = entity['coverage']
        if self.min_area > coverage or self.max_area < coverage:
            return False
        return True


    def _process_sparse_frame(self, frame: dict) -> None:
        """
        Filter entities based on rules:
        - Always keep hands and gloves (not the category 60 glove)
        - Only keep object-in-hand that:
            - Not in black-list categories and items
            - Have reasonable size based on its mask or bbox area
            Both are in config file
            
        frame = {
            'name': name of the image, in format Pxx_xxx_frame_xxxxxxxxxx
            'subsequence': subsequence name that the frame belongs to
            'image_path': absolute path to the .jpg image
            'type': type of frame, sparse or dense
            'annotations': dict of id and object in a frame
            'left hand': list of id of left hand or glove on left hand
            'left object': id of object in contact with left hand
            'right hand': list of id of right hand or glove on right hand
            'right object': id of object in contact with right hand
            'relations': list of (hand/glove_id, object_id) where object is the in-contact object
        }
        """
        # Add fields to frame
        # frame['type'] = 'sparse'
        frame['left hand'] = []
        frame['left object'] = (None, None) # default, (object_id, object_name)
        frame['right hand'] = []
        frame['right object'] = (None, None)
        frame['relations'] = []
        
        # Convert frame's entities list to dictionary for quick query, and add area coverage
        entities = {}
        for entity in frame['annotations']:
            entity['coverage'] = calculate_segment_area(entity['segments'], self.resolution, self.mode) / np.prod(self.resolution)
            entities[entity['id']] = entity
            
        updated_annotations = {}
        
        # Extract hand and object relation in each frame
        
        for entity_id, entity in entities.items():
            
            object_id = entity.get('in_contact_object')
            if entity['class_id'] in VISORData.HAND_IDS:
                frame[entity['name']].append(entity_id) # add hand id to 'left hand' or 'right hand'
                # check if there is an object in hand
                check_object = (object_id not in VISORData.INVALID_CONTACTS
                                and object_id in entities) # for P06_13_frame_0000002608 

                if check_object:
                    # check if the object in hand is a glove on the same hand
                    check_glove = ('on_which_hand' in entities[object_id] 
                                    and entities[object_id]['on_which_hand'] in VISORData.GLOVE_VALUES # have valid values
                                    and entity['name'] in entities[object_id]['on_which_hand']) # hand itself is in list of glove's on_which_hand

                    if not check_glove:
                        # hand contacts with normal object
                        if object_id and self._filter_object(entities[object_id]):
                            frame.setdefault('relations', []).append((entity_id, object_id))
                            frame[entity['name'].replace('hand', 'object')] = (object_id, entities[object_id]['name'])
                    # in case check_glove is true, the next block will handle it
                
            elif 'on_which_hand' in entity and entity['on_which_hand'] in VISORData.GLOVE_VALUES:
                # cover all gloves that is on hand(s)                    
                if len(entity['on_which_hand']) >= 2: # glove's on both hands
                    frame.setdefault('left hand', []).append(entity_id)
                    frame.setdefault('right hand', []).append(entity_id)
                    if object_id and self._filter_object(entities[object_id]): # both hands -> glove -> object (rare case)
                        frame['left object'] = (object_id, entities[object_id]['name'])
                        frame['right object'] = (object_id, entities[object_id]['name'])
                else:
                    side = entity["on_which_hand"][0].split(" ")[0]
                    frame.setdefault(f'{side} hand', []).append(entity_id)
                    if object_id not in VISORData.INVALID_CONTACTS and self._filter_object(entities[object_id]):
                        frame[f'{side} object'] = (object_id, entities[object_id]['name'])
                    
                frame['relations'].append((entity_id, object_id)) # TODO: do we still need relation?
                
            else:
                # normal objects, filter based on object filter
                if not self._filter_object(entity):
                    continue # skip the not passed objects (not add into updated_annotations)
            
            updated_annotations[entity_id] = entity
        
        frame['annotations'] = updated_annotations
        
        return frame
        
    def _sample_frames(self, subseq):
        """
        Rough ideas:
            [] Only preprocessing dense annotation, check if associated frames exists
        """
        
        raise NotImplementedError
    
    def _load_dense_annotation(self, sequence_name: str) -> None:
        """
        Load dense annotation based on input sequence name in format (Pxx_xxx_seq_xxxxx)
        """
        dense_anno_template = os.path.join(self.dense_root, self.subset, '{}_interpolations', '{}_interpolations.json')
        name = sequence_name.split('_seq_')[0]
        if name != self.dense_anno[0]:
            json_file = dense_anno_template.format(name, name)
            if os.path.isfile(json_file):    
                with open(json_file, 'r') as f:
                    data = json.load(f)
            else:
                logging.error(f'{json_file} not found')
            
            dense_frames = {}
            for frame in data['video_annotations']:
                video_name_path = frame['image']['image_path'].split('_')[0]
                frame['image_path'] = os.path.join(self.dense_img_root, self.subset, video_name_path, frame['image']['image_path'].replace('.png', '.jpg')) # TODO: check file exist it when have dense anno folder
                
                if not os.path.isfile(frame['image_path']):
                    continue
                
                first_frame = frame['image']['interpolation_start_frame'].replace('.png', '')
                end_frame = frame['image']['interpolation_end_frame'].replace('.png', '')
                frame['name'] = frame['image']['name'].replace('.png', '')
                if frame['name'] in [first_frame, end_frame]:
                    frame['type'] = 'filtered_sparse'
                else:
                    frame['type'] = 'dense'
                
                frame['start'], frame['end'] = first_frame, end_frame
                del frame['image']
                
                # Convert list of objects to dict with key is object name
                updated_annotations = {}
                for entity in frame['annotations']:
                    if entity['name'] in updated_annotations:
                        logging.error(f'Duplicated entity in {frame["name"]}')
                    updated_annotations[entity['name']] = entity
                frame['annotations'] = updated_annotations
                
                dense_frames.setdefault((first_frame, end_frame), []).append(frame)
            self.dense_anno = (name, dense_frames) 
        
    
    def get_subsequence(self, augment: bool = False):        
        for name, sparse_frames in self.subsequences.items():
            dense_frames = {}
            trajectory = {'left': [], 'right': []}
            logging.info(f'Subsequence {name}')
            
            # Load Sparse annotation first
            for idx, frame in enumerate(sorted(sparse_frames, key = lambda k: k['name'])):
                # Process each frame: dict, sparse_frames: list
                logging_msg = f'Frame #{idx}: {[obj["name"] for obj in frame["annotations"]]} -> '
                
                sparse_frames[idx] = self._process_sparse_frame(frame)
                logging_msg += str([obj["name"] + ": " + str(round(obj["coverage"], 5)) for obj in sparse_frames[idx]["annotations"].values()])
                logging.info(logging_msg)
                
                # Extract object-in-hand
                trajectory['left'].append(frame['left object'][-1])
                trajectory['right'].append(frame['right object'][-1])
                
            if augment:
                sparse_names = [frame['name'] for frame in sparse_frames]
                self._load_dense_annotation(name)
                for i in range(1, len(sparse_names)):
                    key = (sparse_names[i-1], sparse_names[i])
                    # note: not all the sparse frame pairs have corresponding dense annotations
                    dense_frames[key] = self.dense_anno[1].get(key, []) # TODO: Consider sampling from here?
            
            subseq = {'name': name, 
                    'frames': {'sparse': sparse_frames, 'dense': dense_frames}, 
                    'trajectories': trajectory, 
                    }
            
            yield subseq
    
    def get_subsequence_names(self):
        return list(self.subsequences.keys())
    
    def __getitem__(self, name):
        """
        Get raw annotation based on subseq namex
        """
        frames = self.subsequences[name]
        for idx in range(len(frames)):
            frames[idx] = self._process_sparse_frame(frames[idx])
            
        return frames
    
    # For debugging
    @staticmethod
    def draw_frame(frame):
        # Draw a mask and overlay its on images
        colors = [
                    (255, 0, 0),    # red
                    (0, 0, 255),    # blue
                    (0, 255, 0),    # green
                    (128, 0, 0),    # maroon
                    (255, 255, 0),  # yellow
                    (0, 255, 0),    # lime
                    (255, 0, 255),  # fuchsia
                    (0, 255, 255),  # aqua
                    (128, 0, 128),  # purple
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
        cv2.imwrite(os.path.join('./debugs', file_name), overlay)


if __name__ == '__main__':
    SPARSE_ROOT = '../GroundTruth-SparseAnnotations'
    DENSE_ROOT = '../Interpolations-DenseAnnotations'
    DENSE_IMG_ROOT = '../rgb_frames'
    
    SUBSET = 'val'
    
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check config validity
    assert config['coverage_filter']['mode'] in ['bbox', 'mask'], 'Coverage mode must be either bbox or mask'
    assert 0 <= config['coverage_filter']['max_area'] <= 1, 'Max area coverage must be in range [0, 1]'
    assert 0 <= config['coverage_filter']['min_area'] <= 1, 'Min area coverage must be in range [0, 1]'

    data = VISORData(config, SPARSE_ROOT, SUBSET, DENSE_ROOT, DENSE_IMG_ROOT)
    
    sequence = list(data.get_subsequence(True))



