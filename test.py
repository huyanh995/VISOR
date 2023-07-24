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
from glob import glob
import shutil

# from utils import process_poly, segment_iou
# from visor_data import VISORData

import utils
import visor_data

logging.basicConfig(level=logging.ERROR)

class VISORFormatter:
    """
    Output refined VISOR data to DAVIS or YTVOS format.
    Support adding sampled dense frames.
    """
    Frame = namedtuple('Frame', ['type', 'frame', 'mask', 'boundary_mask'])
    # Davis palette
    PALETTE = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
    PALETTE[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                            [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                            [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                            [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                            [0, 64, 128], [128, 64, 128]]

    def __init__(self,
                 data_config: dict,
                 out_root: str,
                 target_res: tuple = (854, 480)) -> None:

        self.config = VISORFormatter._load_config()
        self.data = visor_data.VISORData(self.config, **data_config)
        self.src_res = self.data.resolution
        self.subset = self.data.subset
        self.out_root = out_root

        self.target_res = target_res

        self.relation_mapping = {} # to record the hand-object relation in each frame
        self.color_mapping = {}
        self.sparse_mapping = {}

        self.meta = {'videos': {}} # to record the metadata of each frame, similar to YTVOS

        # for collecting stats
        self.stats = {'num_valid_subseqs': 0,
                    'num_invalid_subseqs': 0,
                    'num_frames': 0,
                    'num_entities': 0,
                    }

        self.coverages_stats = {} # compute coverage of each object names

    def generate(self,
                 output_format: str,
                 augment: bool = False,
                 include_hand: bool = False,
                 reserve_hand: bool = False,
                 include_boundary: bool = False) -> None:
        """Main function to generate dataset in DAVIS or YTVOS format

        Args:
            format (str): data format, either DAVIS or YTVOS
            augment (bool, optional): using dense annotation or not. Defaults to False.
            include_hand (bool, optional): _description_. Defaults to False.
            reserve_hand (bool, optional): _description_. Defaults to False.
        """
        output_format = output_format.lower()
        assert output_format in ['davis', 'ytvos', 'both'], 'only support DAVIS and YTVOS (YouTube-VOS) data format'
        self.augment = augment
        self.include_hand = include_hand
        self.reserve_hand = reserve_hand
        self.include_boundary = include_boundary

        self._prepare_out_dir(output_format)

        for subseq in tqdm(self.data.get_subsequence(augment=augment), total = len(self.data.subsequences)):
            if not self._check_subsequence(subseq):
                self.stats['num_invalid_subseqs'] += 1
                continue
            self.stats['num_valid_subseqs'] += 1
            self.relation_mapping[subseq['name']] = {}

            object_to_color = {}
            if self.reserve_hand:
                object_to_color['left hand'] = 1
                object_to_color['right hand'] = 2

            sparse_frames = []
            for idx, frame in enumerate(subseq['frames']['sparse']):
                mask, boundary_mask, object_to_color, updated_anno = self._process_sparse_frames(frame, object_to_color)
                frame.update(updated_anno)
                sparse_frames.append(VISORFormatter.Frame('sparse', frame, mask, boundary_mask))

            if self.augment:
                combined_frames = [sparse_frames[0]]
                for idx in range(1, len(sparse_frames)):
                    # Sample dense frames based on object name in sparse frames
                    prev_frame = sparse_frames[idx - 1].frame
                    curr_frame = sparse_frames[idx].frame
                    combined_frames += self._sample_dense_frames(subseq['frames']['dense'],
                                                                 prev_frame,
                                                                 curr_frame,
                                                                 object_to_color)
                frames = combined_frames
            else:
                frames = sparse_frames

            # Generate data based on format
            self._prepare_out_dir(output_format)
            self._generate_frames(output_format, subseq['name'], frames, object_to_color)

        # Aggregate meta files and stats
        self._write_meta_file()
        self._print_stats()

    def _write_meta_file(self) -> None:
        """Aggregate meta file
        """
        file_names = glob(os.path.join(self.out_path['Root'], 'temp', '*.json'))
        file_names.sort(key=os.path.getctime) # sort based on process time

        if os.path.isfile(self.out_path['meta']):
            # There is a meta file already, probably from previous run
            with open(self.out_path['meta'], 'r') as f:
                meta = json.load(f)
        else:
            meta = {'videos': {}}

        for file_name in file_names:
            subseq = os.path.basename(file_name).split('.')[0]
            with open(file_name, 'r') as f:
                data = json.load(f)

            meta['videos'][subseq] = data

        print(f'Writing meta.json file to {self.out_path["meta"]}...')
        with open(self.out_path['meta'], 'w') as fs:
            json.dump(meta, fs, indent=4)

        # Write stats
        stats_path = self.out_path['meta'].replace('meta', 'stats')
        print(f'Writing stats.json file to {stats_path}...')
        with open(stats_path, 'w') as fs:
            stats = {'config': {'augment': self.augment,
                                'include_hand': self.include_hand,
                                'reserve_hand': self.reserve_hand,
                                'include_boundary': self.include_boundary},
                     'stats': self.stats}
            json.dump(stats, fs, indent=4)

        # Delete temp folder
        if os.path.isfile(self.out_path['meta']):
            shutil.rmtree(os.path.join(self.out_path['Root'], 'temp'))


    def _sample_dense_frames(self, dense_dict: dict, prev: dict, curr: dict, object_to_color: dict) -> list[Frame]:
        """_summary_

        Args:
            dense_dict (dict): dictionary of all dense frames in a sequence (more than 1 subseq)
            prev (dict): start interpolation frame
            curr (dict): end interpolation frame
            object_to_color (dict): mapping dict, only select object in this dict

        Returns:
            list[Frame]: list of dense frames
        """
        key = (prev['name'], curr['name'])
        dense_frames = dense_dict[key]
        # Step 1: filter out frames that do not have object in object_to_color


        print('DEBUG')



        return None

    def _check_subsequence(self, subseq: dict) -> dict:
        """Process subsequence from VISORData:
        - Check if a subsequence has enough length and valid trajectory

        Args:
            subseq (dict): a dictionary of frames

        Returns:
            dict: filtered subsequence
        """
        sparse_frames = subseq['frames']['sparse']
        if len(sparse_frames) < self.config['max_length']:
            return False

        left_traj = subseq['trajectories']['left']
        right_traj = subseq['trajectories']['right']

        # Check if a subsequence has valid trajectory
        # an object is held on a hand at least config's # of frames
        is_valid = (VISORFormatter.is_trajectory(left_traj, self.config['window'])
                    or VISORFormatter.is_trajectory(right_traj, self.config['window']))

        return is_valid


    def _process_sparse_frames(self,
                               frame: dict,
                               object_to_color: dict) -> None:
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
        """
        mask = np.zeros(self.src_res, dtype=np.uint8)
        boundary_mask = np.zeros((*self.src_res, 3), dtype=np.uint8) # left, right and background
        # Get next color for new object, start from 1
        next_color = sorted(object_to_color.values())[-1] + 1 if object_to_color else 1

        # Updated annotation based on object name for dense annotation propagation
        updated_anno = {'left hand': [],
                    'left object': None,
                    'right hand': [],
                    'right object': None,
                    'relations': [],
                    'coverage': {}}

        mask_storage = []

        for idx, side in enumerate(['left', 'right']):
            #===== Process hand first =================
            hand_ids = frame[f'{side} hand'] # can be hand and glove
            if len(hand_ids) == 0:
                # No {side} hand in this frame, skip also the {side} object
                if frame[f'{side} object'][0]:
                    logging.error(f'Subseq {frame["subsequence"]} {side} side: object but no hand')
                continue

            if len(hand_ids) == 1:
                hand_polygons = utils.process_poly(frame['annotations'][hand_ids[0]]['segments'])

                hand_coverage = frame['annotations'][hand_ids[0]]['coverage']
                updated_anno[f'{side} hand'].append(frame['annotations'][hand_ids[0]]['name'])

            else:
                # Glove on hand, will merge into one mask
                ls_segments = [frame['annotations'][hand_id]['segments'] for hand_id in hand_ids]
                ls_names = [frame['annotations'][hand_id]['name'] for hand_id in hand_ids]
                hand_polygons = VISORFormatter._merge_polygons(ls_segments)

                hand_coverage = sum([frame['annotations'][hand_id]['coverage'] for hand_id in hand_ids])
                updated_anno[f'{side} hand'] = ls_names


            if self.include_hand:
                hand_color = object_to_color.get(f'{side} hand', next_color)
                mask_storage.append((hand_color, hand_polygons))
                object_to_color[f'{side} hand'] = hand_color
                next_color = max(next_color, hand_color + 1)

            hand_class = 300 if side == 'left' else 301
            updated_anno['coverage'][f'{side} hand'] = (hand_class, hand_coverage)

            #===== Process object in contact with hand ============
            if (object_id := frame[f'{side} object'][0]):
                object_name = frame['annotations'][object_id]['name']
                object_class = frame['annotations'][object_id]['class_id']
                object_polygons = utils.process_poly(frame['annotations'][object_id]['segments'])

                filtered_object_polygons, _boundary_mask, object_coverage = self._filter_object_segments(hand_polygons,
                                                                                                        object_polygons,
                                                                                                        iterations = 5)

                object_color = object_to_color.get(object_name, next_color)
                mask_storage.append((object_color, filtered_object_polygons))
                object_to_color[object_name] = object_color
                next_color = max(next_color, object_color + 1)

                updated_anno[f'{side} object'] = object_name
                updated_anno['coverage'][object_name] = (object_class, object_coverage)

                # Add to relation mapping
                self.relation_mapping[frame['subsequence']].setdefault(frame['name'], []) \
                                                        .append([hand_color, object_color])

                #===== Process contact boundary mask if needed ========
                if self.include_boundary:
                        boundary_mask[:, :, idx] = np.where(_boundary_mask > 0, 128, 0).astype(np.uint8)

        # Fill mask
        for color, polygons in mask_storage:
            cv2.fillPoly(mask, polygons, (color, color, color))

        # Update relations to name
        for hand_id, object_id in frame['relations']:
            updated_anno['relations'].append([frame['annotations'][hand_id]['name'],
                                            frame['annotations'][object_id]['name']])

        return mask, boundary_mask, object_to_color, updated_anno


    def _generate_frames(self,
                        out_format: str,
                        name: str,
                        frames: list[Frame],
                        object_to_color: dict) -> None:

        """From list of frames, generate to DAVIS format
        - Resize image, masks to target resolution
        - Write masks to Annotations folder using defined palette.
        - Write contact boundaries to Boundaries folder.
        - Write RGB images to JPEGImages folder.

        Args:
            out_format (str): output format, either DAVIS or YTVOS
            name (str): subsequence name, to create a folder
            frames (list[Frame]): list of Frames
            object_to_color (dict): mapping object to color
        """
        # Create a folder for each subsequence
        file_name = '{:05}'
        img_path = os.path.join(self.out_path['JPEGImages'], name, file_name + '.jpg')
        mask_path = os.path.join(self.out_path['Annotations'], name, file_name + '.png')
        boundary_path = os.path.join(self.out_path['Boundaries'], name, file_name + '_boundary.png')

        os.makedirs(os.path.dirname(img_path))
        os.makedirs(os.path.dirname(mask_path))
        os.makedirs(os.path.dirname(boundary_path))

        meta_path = os.path.join(self.out_path['Root'], 'temp', f'{name}.json')
        os.makedirs(os.path.dirname(meta_path), exist_ok=True) # Won't do nothing if temp is exist

        meta = {'objects': {str(k): {'category': None, 'frames': []} for k in object_to_color.values()},
                'frames': {},
                'coverage': {},
                'object_to_color' : None,
                }

        color_to_object = {v: k for k, v in object_to_color.items()}

        for idx, frame in enumerate(frames):
            img = cv2.imread(frame.frame['image_path'])
            img = cv2.resize(img, self.target_res, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(frame.mask, self.target_res, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

            # Write image and mask
            cv2.imwrite(img_path.format(idx), img)
            self._write_mask(mask_path.format(idx), mask)

            if self.include_boundary:
                boundary_mask = cv2.resize(frame.boundary_mask, self.target_res, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                Image.fromarray(boundary_mask, 'RGB').save(boundary_path.format(idx))

            # Update meta data
            for i in np.unique(mask):
                if i == 0: # background
                    continue
                meta['objects'][str(i)]['category'] = frame.frame['coverage'][color_to_object[i]][0]
                meta['coverage'].setdefault(color_to_object[i], {})[file_name.format(idx)] = frame.frame['coverage'][color_to_object[i]][1]
                meta['objects'][str(i)]['frames'].append(file_name.format(idx))

            tmp = {}
            tmp['left hand'], tmp['left object'] = frame.frame['left hand'], frame.frame['left object']
            tmp['right hand'], tmp['right object'] = frame.frame['right hand'], frame.frame['right object']

            for k, v in frame.frame['relations']:
                if k not in object_to_color:
                    k = 'left hand' if k in frame.frame['left hand'] else 'right hand'
                tmp.setdefault('relations', []).append((object_to_color[k], object_to_color[v]))
            meta['frames'][file_name.format(idx)] = tmp
            meta['object_to_color'] = object_to_color
            # Update stats
            self.stats['num_entities'] += len(np.unique(mask)) - 1 # exclude background
            self.stats['num_frames'] += 1

        # Write txt meta data file (specialize for DAVIS)
        if out_format == 'davis':
            imageset_path = os.path.join(self.out_path['ImageSets'], f'{self.subset}.txt')
            with open(imageset_path, 'a') as fs:
                fs.write(name + '\n')

        # Write temporary metadata for each subsequence
        with open(meta_path, 'w') as fs:
            json.dump(meta, fs, indent=4)


    def _generate_ytvos(self, name, frames: list[Frame]) -> None:

        raise NotImplementedError

    def _write_mask(self, out_path: str, mask: np.array) -> None:
        """Write mask into png file with palette, with resized resolution
        DAVIS palette from VISOR repo.

        Args:
            out_path (str): out directory for mask
            mask (np.array): mask in numpy array
        """
        assert len(mask.shape) < 4 or mask.shape[0] == 1
        # First resize mask to target resolution
        mask = cv2.resize(mask, self.target_res, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # Write mask
        mask = Image.fromarray(mask, 'P')
        mask.putpalette(VISORFormatter.PALETTE.ravel())
        mask.save(out_path)

    def _update_mapping(self) -> None:
        with open(os.path.join(self.out_path['Root'], 'color_mapping.json'), 'w') as f:
                json.dump(self.color_mapping, f)

        with open(os.path.join(self.out_path['Root'], 'hand_relation.json'), 'w') as f:
            json.dump(self.relation_mapping, f)

        with open(os.path.join(self.out_path['Root'], 'sparse_mapping.json'), 'w') as f:
            json.dump(self.sparse_mapping, f)


    def _print_stats(self) -> None:
        """
        Write dataset stats to file and print to console
        """
        # Print stats
        if self.augment:
            print(f'Stats for refined {self.subset} set with dense frames')
        else:
            print(f'Stats for refined {self.subset} set')
        print(f'Total subsequences: {self.stats["num_valid_subseqs"]}')
        print(f'Total frames: {self.stats["num_frames"]}')
        print(f'Total objects: {self.stats["num_entities"]}')
        print('-' * 20)


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
                            raw_info['annotations'][name] = utils.process_poly(segment)

                else:
                    hand_polygons = utils.process_poly(frame['annotations'][hand_ids[0]]['segments']) # only 1 element in the list
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
                object_polygons = utils.process_poly(frame['annotations'][object_id]['segments'])

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


    def _filter_object_segments(self, hand_polygons, object_polygons, iterations):
        """
        Filter object segments not in contact with hand based on intersection over mask dilation
        Args:
            hand_polygons (List[np.array]): merged into one mask
            object_polygons: (List[np.array]): each array is a polygons

        Returns:
            List[np.array]: filtered object polygons
            np.array: contact boundary mask
        """
        hand_mask = np.zeros(self.src_res, dtype=np.uint8)
        cv2.fillPoly(hand_mask, hand_polygons, (1, 1, 1)) # Binary mask
        hand_mask = VISORFormatter._dilate(hand_mask, iterations)

        res = []
        object_coverage = 0
        contact_mask = np.zeros(self.src_res, dtype=np.uint8)

        for polygons in object_polygons:
            object_mask = np.zeros(self.src_res, dtype=np.uint8)
            cv2.fillPoly(object_mask, [polygons], (1, 1, 1))
            dil_object_mask = VISORFormatter._dilate(object_mask, iterations)

            # Check for overlapping between dilated hand and object masks
            _mask = np.bitwise_and(hand_mask, dil_object_mask)
            overlap_score = np.sum(_mask)
            if overlap_score > self.config['instance_filter']['overlap_threshold']:
                res.append(polygons)
            contact_mask += _mask
            object_coverage += np.sum(object_mask)

        return res, contact_mask, object_coverage / np.prod(self.src_res)


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


    def _prepare_out_dir(self, out_format: str) -> None:
        """Generate output directory for DAVIS or YTVOS format

        Args:
            format (str): data format, either DAVIS or YTVOS
        """
        resolution = f'{self.target_res[-1]}p'
        folder_name = f'{self.data.name}_{self.data.year}' # 'VISOR_2022' by default
        if out_format.lower() == 'davis':
            folder_name = folder_name + '_DAVIS'
            self.out_path = {
                'Annotations': os.path.join(self.out_root, folder_name, 'Annotations', resolution),
                'ImageSets': os.path.join(self.out_root, folder_name, 'ImageSets', self.data.year),
                'JPEGImages': os.path.join(self.out_root, folder_name, 'JPEGImages', resolution),
                'Boundaries': os.path.join(self.out_root, folder_name, 'Boundaries', resolution),
                'meta': os.path.join(self.out_root, folder_name, 'meta.json'),
                'Root': os.path.join(self.out_root, folder_name),
            }

            try:
                # FIXME: remove exist_ok later, this is for testing purposes only
                os.makedirs(self.out_path['Annotations'], exist_ok=True)
                os.makedirs(self.out_path['ImageSets'], exist_ok=True)
                os.makedirs(self.out_path['JPEGImages'], exist_ok=True)
                os.makedirs(self.out_path['Boundaries'], exist_ok=True)

            except FileExistsError:
                print("VISOR exists, please choose different location!")
                exit()

        elif out_format.lower() == 'ytvos':
            folder_name = folder_name + '_YTVOS'
            self.out_path = {
                'Annotations': os.path.join(self.out_root, folder_name, self.subset, 'Annotations'),
                'JPEGImages': os.path.join(self.out_root, folder_name, self.subset, 'JPEGImages'),
                'Boundaries': os.path.join(self.out_root, folder_name, self.subset, 'Boundaries'),
                'meta': os.path.join(self.out_root, folder_name, self.subset, 'meta.json'),
                'Root': os.path.join(self.out_root, folder_name, self.subset),
            }
            try:
                os.makedirs(self.out_path['Annotations'], exist_ok=True)
                os.makedirs(self.out_path['JpegImages'], exist_ok=True)
                os.makedirs(self.out_path['GTMasks'], exist_ok=True)

            except FileExistsError:
                print("VISOR exists, please choose different location!")
                exit()


    @staticmethod
    def _dilate(mask, iterations: int = 5):
        mask = ndimage.binary_dilation(mask, iterations = iterations).astype(mask.dtype)
        return mask


    @staticmethod
    def _merge_polygons(ls_segments: list):
        res = []
        for segment in ls_segments:
            res += utils.process_poly(segment)
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
        parser.add_argument('-s', '--set', type=str, help='train or val', default='train')
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
        'dense_img_dir' : args.dense_img,
        'subset' : args.set,
        }

    out_root = '.'
    # data_formatter = VISORFormatter(data_config, out_root, augment=args.dense, include_hand=args.hand)
    # DEBUG
    data_formatter = VISORFormatter(data_config, out_root)
    data_formatter.generate('DAVIS',
                            include_hand=True,
                            reserve_hand=True,
                            augment=False,
                            include_boundary=True)
