"""
Huy Anh Nguyen, CS Stony Brook University
Created on Apr 14, 2023 - 8:07 PM, EDT
"""

import os
import argparse
import json
import logging
import math
from collections import namedtuple

import yaml
from PIL import Image
import numpy as np
import cv2
from scipy import ndimage
from tqdm import tqdm

from utils import process_poly, segment_iou
from visor_data import VISORData

logging.basicConfig(level=logging.ERROR)

class VISORFormatter:
    """
    Output refined VISOR data to DAVIS or YTVOS format.
    Support adding sampled dense frames.
    """
    Frame = namedtuple('Frame', ['type', 'image_path', 'mask', 'raw_info'])

    def __init__(self,
                 data_config: dict,
                 out_root: str,
                 data_format: str = 'DAVIS',
                 resolution: tuple = (854, 480),
                 augment: bool = False,
                 include_hand: bool = False):

        assert data_format in ['DAVIS', 'YTVOS'], 'only support DAVIS and YTVOS (YouTube-VOS) data format'
        self.config = VISORFormatter._load_config()
        self.data = VISORData(self.config, **data_config)

        self.subset = self.data.subset
        self.out_root = out_root
        self.resolution = resolution

        self.data_format = data_format
        self.augment = augment
        self.include_hand = include_hand

        self.relation_mapping = {} # to record the relation mapping in a subsequence
        self.color_mapping = {}
        self.sparse_mapping = {}

        # for collecting stats
        self.stats = {'num_valid_subseqs': 0,
                    'num_invalid_subseqs': 0,
                    'num_frames': 0,
                    'num_entities': 0,
                    'categories': {},
                    }

        self.coverages_stats = {} # compute coverage of each object names


    def generate_davis(self,
                       include_hands: bool = True,
                       reserve: bool = True,
                       include_boundary: bool = True) -> None:

        """Generate VISOR dataset in DAVIS format
        Args:
            include_hands (bool, optional): whether to include hand mask. Defaults to True.
            reserve (bool, optional): whether to reserve the 1 and 2 for left and right hand respectively. Defaults to True.
            include_boundary (bool, optional): whether to include contact boundary mask. Defaults to True.

        Note: reserve hand will violate the DAVIS format (numbers must be consecutive).

        """
        self._prepare_out_dir()
        for subseq in tqdm(self.data.get_subsequence(self.augment), total = len(self.data.subsequences)):
        # for subseq in self.data.get_subsequence():
            subseq = self._process_subseq(subseq)
            if not subseq:
                self.stats['num_invalid_subseqs'] += 1
                continue # invalid subseq

            # Update to stats
            self.stats['num_valid_subseqs'] += 1
            # self.stats['num_frames'] += len(subseq['frames'])

            subseq_name = subseq['name']

            # DAVIS format:
            img_path = os.path.join(self.out_path['JPEGImages'], subseq_name)
            mask_path = os.path.join(self.out_path['Annotations'], subseq_name)
            os.makedirs(img_path, exist_ok=True)
            os.makedirs(mask_path, exist_ok=True)

            sparse_frames = []
            self.relation_mapping[subseq_name] = {}

            object_to_color = {}
            if reserve:
                object_to_color['left hand'] = 1
                object_to_color['right hand'] = 2

            for idx, frame in enumerate(subseq['frames']['sparse']):
                # add to stats
                self.stats['num_entities'] += len(frame['annotations'])

                # Get coverage stats from sparse annotation only
                # for entity in frame['annotations'].values():
                #     self.stats['categories'].setdefault(entity['class_id'], {}) \
                #                             .setdefault(entity['name'], []).append(entity['coverage'])
                # ============

                # file_name = f'{idx:05}'
                # copy and (optionally) resize img
                # img = cv2.imread(frame['image_path'])
                # resized_img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_LINEAR)

                # generate mask
                mask = np.zeros((1080, 1920), dtype=np.uint8)

                # note: _process_sparse_mask will merge hand and object on hand together as one.
                # thus, becareful when using mask to propagate into dense frames.
                mask, object_to_color, raw_info = self._process_sparse_mask(subseq_name,
                                                                            frame,
                                                                            object_to_color,
                                                                            reserve)

                raw_info['name'] = frame['name']
                # resize to (854, 480) to match with dense annotation
                resized_mask = cv2.resize(mask, (854, 480), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                sparse_frames.append(VISORFormatter.Frame('sparse', frame['image_path'], resized_mask, raw_info))

            if self.augment:
                combined_frames = [sparse_frames[0]]
                for idx in range(1, len(sparse_frames)):
                    prev_info = sparse_frames[idx - 1].raw_info
                    curr_info = sparse_frames[idx].raw_info
                    dense_key = (prev_info['name'], curr_info['name'])
                    combined_frames += self._sample_dense_frames(subseq['frames']['dense'][dense_key], prev_info, curr_info, object_to_color)
                    combined_frames.append(sparse_frames[idx])
            else:
                combined_frames = sparse_frames

            self.stats['num_frames'] += len(combined_frames)

            # Write image and mask
            for idx, frame in enumerate(combined_frames):
                if frame.mask.max() > 0:
                    print('DEBUG')
                file_name = f'{idx:05}'
                img = cv2.imread(frame.image_path)
                resized_img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_LINEAR)
                resized_mask = cv2.resize(frame.mask, self.resolution, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

                # Write which frame is sparse to a dict
                if frame.type == 'sparse':
                    self.sparse_mapping.setdefault(subseq_name, []).append(file_name)

                cv2.imwrite(os.path.join(img_path, file_name + '.jpg'), resized_img)
                VISORFormatter.imwrite_mask(os.path.join(mask_path, file_name + '.png'), resized_mask)


            # Update data mapping and relation mapping
            self.color_mapping[subseq_name] = {v: k for k, v in object_to_color.items()}
            # Append subseq name to txt file
            with open(os.path.join(self.out_path['ImageSets'], f'{self.subset}.txt'), 'a') as f:
                f.write(subseq_name + '\n')
            self._update_mapping()

        # Print stats
        self._write_stats()

    def _update_mapping(self) -> None:
        with open(os.path.join(self.out_path['Root'], 'color_mapping.json'), 'w') as f:
                json.dump(self.color_mapping, f)

        with open(os.path.join(self.out_path['Root'], 'hand_relation.json'), 'w') as f:
            json.dump(self.relation_mapping, f)

        with open(os.path.join(self.out_path['Root'], 'sparse_mapping.json'), 'w') as f:
            json.dump(self.sparse_mapping, f)


    def _write_stats(self) -> None:
        """
        Write dataset stats to file and print to console
        """
        # Write stats to file
        with open(os.path.join(self.out_path['Root'], f'{self.subset}_stats.json'), 'w') as f:
            # process categories
            for class_id, class_object in self.stats['categories'].items():
                for name, ls_coverage in class_object.items():
                    self.stats['categories'][class_id][name] = round(sum(ls_coverage) / len(ls_coverage), 6)
            json.dump(self.stats, f)

        # Print stats
        if self.augment:
            print(f'Stats for refined {self.subset} set with dense frames')
        else:
            print(f'Stats for refined {self.subset} set')
        print(f'Total subsequences: {self.stats["num_valid_subseqs"]}')
        print(f'Total frames: {self.stats["num_frames"]}')
        print(f'Total objects: {self.stats["num_entities"]}')
        print('-' * 20)


    def generate_ytvos(self):
        # TODO: remember to extract sparse frames based on frame['type']
        raise NotImplementedError


    def _process_sparse_mask(self,
                             subseq_name: str,
                             frame: dict,
                             object_to_color: dict,
                             reserve: bool):
        """Process sparse mask for each frame in a subsequence.
        Update object_to_color after each frame.

        Args:
            subseq_name (str): name of subsequence
            frame (dict): dictionary of frame metadata and its object annotations
            object_to_color (dict): dictionary of mapping object to color
            reserve (bool): whether to reserve 1 and 2 for left and right hand respectively

        Returns:
            mask: np.array of mask
            object_to_color: updated object_to_color
            raw_info: raw information for each frame (passthrough)
        """


        mask = np.zeros((1080, 1920), dtype=np.uint8)
        # Get the next color, start from 1
        next_color = sorted(object_to_color.values())[-1] + 1 if object_to_color else 1
        raw_info = {'annotations': {},
                    'left hand': [],
                    'left object': None,
                    'right hand': [],
                    'right object': None} # each object polygons and hand-object relations for dense propagation

        for side in ['left', 'right']:
            if (object_id := frame[f'{side} object'][0]): # there should be only one value here
                # Should be a same side hand here
                hand_ids = frame[f'{side} hand']

                if len(hand_ids) == 0:
                    logging.error(f'Subseq {subseq_name} {side} side: object but no hand')

                if len(hand_ids) > 1:
                        ls_segments = [frame['annotations'][hand_id]['segments'] for hand_id in hand_ids]
                        ls_names = [frame['annotations'][hand_id]['name'] for hand_id in hand_ids]
                        hand_polygons = VISORFormatter._merge_polygons(ls_segments)
                        hand_coverage = sum([frame['annotations'][hand_id]['coverage'] for hand_id in hand_ids])

                        # add hand and glove separatedly into raw_info
                        # because dense annotations do not have hand-object relation
                        raw_info[f'{side} hand'] = ls_names
                        for name, segment in zip(ls_names, ls_segments):
                            raw_info['annotations'][name] = process_poly(segment)

                else:
                    hand_polygons = process_poly(frame['annotations'][hand_ids[0]]['segments']) # only 1 element in the list
                    hand_coverage = frame['annotations'][hand_ids[0]]['coverage']

                    hand_name = frame['annotations'][hand_ids[0]]['name']
                    raw_info[f'{side} hand'].append(hand_name)
                    raw_info['annotations'][hand_name] = hand_polygons

                if self.include_hand: # add hand polygons into mask
                    hand_color = object_to_color.get(f'{side} hand', next_color)
                    cv2.fillPoly(mask, hand_polygons, (hand_color, hand_color, hand_color))
                    object_to_color[f'{side} hand'] = hand_color
                    next_color = max(next_color, hand_color + 1)
                    hand_class = 300 if side == 'left' else 301
                    # merged hand and glove as hand class.
                    self.stats['categories'].setdefault(hand_class, {})\
                                            .setdefault(f'{side} hand', []).append(hand_coverage)

                object_name = frame['annotations'][object_id]['name']
                object_class = frame['annotations'][object_id]['class_id']
                object_polygons = process_poly(frame['annotations'][object_id]['segments'])

                raw_info[f'{side} object'] = object_name
                raw_info['annotations'][object_name] = object_polygons

                filtered_object_polygons = self._filter_object_segments(hand_polygons, object_polygons, mask.shape, iterations = 5)

                object_color = object_to_color.get(object_name, next_color)
                cv2.fillPoly(mask, filtered_object_polygons, (object_color, object_color, object_color))
                object_to_color[object_name] = object_color
                next_color = max(next_color, object_color + 1)

                # Recalculate object coverage and add into stats
                object_coverage = np.sum(np.where(mask == object_color, 1, 0)) / np.prod(mask.shape)
                self.stats['categories'].setdefault(object_class, {})\
                                        .setdefault(object_name, []).append(object_coverage)

                # Add to relation mapping
                self.relation_mapping[subseq_name].setdefault(frame['name'], []).append([hand_color, object_color])

        return mask, object_to_color, raw_info


    def _filter_object_segments(self, hand_polygons, object_polygons, img_size, iterations):
        """
        Filter object segments not in contact with hand based on intersection over mask dilation
        hand_polygons: list of np.array(s), merged into one mask
        object_polygons: list of np.array(s), each array is a polygons
        """
        hand_mask = np.zeros(img_size, dtype=np.uint8)
        cv2.fillPoly(hand_mask, hand_polygons, (1, 1, 1)) # Binary mask
        hand_mask = VISORFormatter._dilate(hand_mask, iterations)

        res = []
        for polygons in object_polygons:
            object_mask = np.zeros(img_size, dtype=np.uint8)
            cv2.fillPoly(object_mask, [polygons], (1, 1, 1))
            object_mask = VISORFormatter._dilate(object_mask, iterations)
            # object_mask = VISORFormatter._dilate(object_mask, self.config['instance_filter']['iterations'], iterations)

            # Check for overlapping between dilated hand and object masks
            overlap_score = np.sum(np.bitwise_and(hand_mask, object_mask))
            if overlap_score > self.config['instance_filter']['overlap_threshold']:
                res.append(polygons)

        return res


    def _process_subseq(self, subseq):
        """
        Only keep hand and in-contact object
        """
        # name = subseq['name']
        frames = subseq['frames']['sparse']
        left_trajectory = subseq['trajectories']['left']
        right_trajectory = subseq['trajectories']['right']

        # at least 1 valid traj
        have_valid_traj = (VISORFormatter.is_trajectory(left_trajectory, self.config['window'])
                        or VISORFormatter.is_trajectory(right_trajectory, self.config['window']))

        if len(frames) >= self.config['max_length'] and have_valid_traj:
            # Filter irrelevant objects in each frame
            # Only keep object-in-hand and also hands that have contact with object
            # standalone hands are not acceptable

            for frame in frames: # frames is already sorted
                entities = frame['annotations']
                new_entities = {}
                if (left_object_id := frame.get('left object', [None])[0]):
                    new_entities[left_object_id] = entities[left_object_id]
                    # Get left hand object, at least have 1
                    if self.include_hand:
                        for left_hand_id in frame['left hand']:
                            new_entities[left_hand_id] = entities[left_hand_id]

                if (right_object_id := frame.get('right object', [None])[0]):
                    new_entities[right_object_id] = entities[right_object_id]
                    # Get right hand object, at least have 1
                    for right_hand_id in frame['right hand']:
                        new_entities[right_hand_id] = entities[right_hand_id]

                # record the filtered entities into frame
                frame['entities'] = new_entities

            return subseq

        return None # in case trajectory is not valid


    def _prepare_out_dir(self):
        resolution = f'{self.resolution[-1]}p'
        folder_name = f'{self.data.name}_{self.data.year}' # 'VISOR_2022' by default
        if self.data_format == 'DAVIS':
            self.out_path = {
                'Annotations': os.path.join(self.out_root, folder_name, 'Annotations', resolution),
                'ImageSets': os.path.join(self.out_root, folder_name, 'ImageSets', self.data.year),
                'JPEGImages': os.path.join(self.out_root, folder_name, 'JPEGImages', resolution),
                'Root': os.path.join(self.out_root, folder_name),
            }

            try:
                # FIXME: remove exist_ok later, this is for testing purposes only
                os.makedirs(self.out_path['Annotations'], exist_ok=True)
                os.makedirs(self.out_path['ImageSets'], exist_ok=True)
                os.makedirs(self.out_path['JPEGImages'], exist_ok=True)

            except FileExistsError:
                print("VISOR exists, please choose different location!")
                exit()

        else:
            self.out_path = {
                'Annotations': os.path.join(self.out_root, f'YTVOS_{folder_name}', self.subset, 'Annotations'),
                'JpegImages': os.path.join(self.out_root, f'YTVOS_{folder_name}', self.subset, 'JPEGImages'),
                'GTMasks': os.path.join(self.out_root, f'YTVOS_{folder_name}', self.subset, 'GTMasks'),
            }
            try:
                os.makedirs(self.out_path['Annotations'], exist_ok=True)
                os.makedirs(self.out_path['JpegImages'], exist_ok=True)
                os.makedirs(self.out_path['GTMasks'], exist_ok=True)

            except FileExistsError:
                print("VISOR exists, please choose different location!")
                exit()

    def _sample_dense_frames(self, input_frames, prev_info, curr_info, object_to_color):
        """
        Sample dense frames in between 2 sparse frames (prev_info and curr_info)
        * for hand and glove on hand, on each side, there should be only 1 instance
        * for object in contact with hand, need to handle multi-instance cases based on:
            * IoU with sparse instances (should be enough)
            * distance to respective hand (should be experimental)
        * should use blurry frame detection to filter out frames?
        """
        if len(input_frames) == 0:
            return []
        # Check if generating dense frames is valid
        is_object_valid = (bool(prev_info['left object']) and prev_info['left object'] == curr_info['left object']) \
            or (bool(prev_info['right object']) and prev_info['right object'] == curr_info['right object']) # at least one hand still held the same object

        if not is_object_valid:
            return []

        # Sample frames
        res = []
        start_idx = int(prev_info['name'].split('_')[-1])
        end_idx = int(curr_info['name'].split('_')[-1])
        duration = (end_idx - start_idx) / 60.0 # VISOR was recorded in 60 fps
        num_frames = math.ceil(duration * self.config['dense_sample']['fps']) # include sparse frames on two ends
        dense_frames = [input_frames[i] for i in np.linspace(0, len(input_frames) - 1, num_frames, dtype=int)[1: -1]]

        if dense_frames:
            for side in ['left', 'right']:
                # For each side, check if side hand held the same object.
                # if yes, then propagate that object with (optional) hand across dense frames
                if prev_info[f'{side} object'] == curr_info[f'{side} object']:
                    object_name = prev_info[f'{side} object']
                    hand_names = prev_info[f'{side} hand']

                    # Propagate hand first
                    for frame in dense_frames:
                        frame['new_annotations'] = {}
                        if not object_name in frame['annotations']:
                            # hand held object not in the frame, skip to the next
                            continue

                        hand_polygons = []
                        for name in hand_names:
                            if name in frame['annotations']:
                                # not all dense frames have hand in it
                                # or for hand and glove, maybe only hand in the frame.
                                hand_polygons += process_poly(frame['annotations'][name]['segments'])

                        # if len(hand_polygons) == 0:
                        #     # No hand in the frame, ignore hand-held object
                        #     continue

                        object_polygons = process_poly(frame['annotations'][object_name]['segments'])

                        # filter object polygons the same as sparse annotation
                        if hand_polygons and len(object_polygons) > 1:
                            filtered_object_polygons = self._filter_object_segments(hand_polygons, object_polygons, (480, 854), iterations = 8) # with larger dilute iterations due to noisy segmentataions in dense frames
                        else:
                            filtered_object_polygons = object_polygons
                        if self.include_hand:
                            frame['new_annotations'][f'{side} hand'] = hand_polygons

                        frame['new_annotations'][object_name] = filtered_object_polygons

            # Generate mask and output for the dense frames
            for frame in dense_frames:
                # generate mask based on new_annotations
                mask = np.zeros((480, 854), dtype=np.uint8)
                for object_name, object_polygons in frame['new_annotations'].items():
                    color = object_to_color[object_name]
                    cv2.fillPoly(mask, object_polygons, (color, color, color))

                res.append(VISORFormatter.Frame('dense', frame['image_path'], mask, None)) # Dense frames don't need raw_info

        return res


    @staticmethod
    def _dilate(mask, iterations: int = 5):
        mask = ndimage.binary_dilation(mask, iterations = iterations).astype(mask.dtype)
        return mask


    @staticmethod
    def _merge_polygons(ls_segments: list):
        res = []
        for segment in ls_segments:
            res += process_poly(segment)
        return res


    @staticmethod
    def _load_config(filename = 'config.yml'):
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)

        # Check config validity
        assert config['coverage_filter']['mode'] in ['bbox', 'mask'], 'Coverage mode must be either bbox or mask'
        assert 0 <= config['coverage_filter']['max_area'] <= 1, 'Max area coverage must be in range [0, 1]'
        assert 0 <= config['coverage_filter']['min_area'] <= 1, 'Min area coverage must be in range [0, 1]'

        return config

    @staticmethod
    def imwrite_mask(out_path, mask):
        """
        To write mask
        """
        davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
        davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                                [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                                [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                                [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                                [0, 64, 128], [128, 64, 128]]
        assert len(mask.shape) < 4 or mask.shape[0] == 1
        mask = Image.fromarray(mask, 'P')
        mask.putpalette(davis_palette.ravel())
        mask.save(out_path)

    @staticmethod
    def is_trajectory(traj: list, window: int = 2) -> bool:
        res = False
        for i in range(window, len(traj) + 1):
            tmp_set = set(traj[i - window : i])
            if len(tmp_set) == 1 and tmp_set != set([None]):
                res = True
        return res

if __name__ == '__main__':
    # TODO: Add args
    def parse_args():
        parser = argparse.ArgumentParser(description='VISOR Data generator into DAVIS or YTVOS')
        parser.add_argument('-s', '--set', type=str, help='train or val', default='val')
        parser.add_argument('-sr', '--sparse_root', type=str, help='path to GroundTruth-SparseAnnotations', default='../GroundTruth-SparseAnnotations')
        parser.add_argument('--dense', action='store_true', help='add dense frames')
        parser.add_argument('-dr', '--dense_root', type=str, help='path to Interpolations-DenseAnnotations', default='../Interpolations-DenseAnnotations')
        parser.add_argument('-di', '--dense_img', type=str, help='path to extracted dense frames from EPIC-KITCHEN', default='../rgb_frames')
        parser.add_argument('--hand', action='store_true', help='add hands mask')

        args = parser.parse_args()
        return args

    args = parse_args()

    data_config = {
        'sparse_root' : args.sparse_root,
        'dense_root' : args.dense_root,
        'dense_img_root' : args.dense_img,
        'subset' : args.set,
        }

    out_root = '.'
    # data_formatter = VISORFormatter(data_config, out_root, augment=args.dense, include_hand=args.hand)
    # DEBUG
    data_formatter = VISORFormatter(data_config, out_root, augment=args.dense, include_hand=True)
    data_formatter.generate_davis()

    print("DEBUG")
