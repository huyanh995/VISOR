import numpy as np
from scipy.ndimage import label
import cv2

def mask_to_single_bbox(mask, obj_index):
    # Output bounding box from a 2D mask
    # TODO: if an object have two connected regions (e.g two onions)
    # then using scipy.ndimage.label with proper centrosymmetric matrix is needed
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
    binary_mask = np.where(mask == obj_index, 1, 0)
    # indices = np.argwhere(binary_mask) # find indices of non-zero elements
    # x_min, y_min = np.min(indices, axis = 0)
    # x_max, y_max = np.max(indices, axis = 0)
    
    indices = binary_mask.nonzero() # find indices of non-zero elements
    x_min, x_max = min(indices[0]), max(indices[0])
    y_min, y_max = min(indices[1]), max(indices[1])
    
    return (x_min, y_min, x_max, y_max)


def mask_to_multi_bbox(mask, obj_index):
    # EXPERIMENTAL: convert mask that has multiple regions of an object
    # to a list of bboxes
    binary_mask = np.where(mask == obj_index, 1, 0)
    labeled_mask, num_labels = label(binary_mask)
    bboxes = []
    
    for i in range(1, num_labels + 1):
        indices = (labeled_mask == i).nonzero()
        x_min, x_max = min(indices[0]), max(indices[0])
        y_min, y_max = min(indices[1]), max(indices[1])
        bboxes.append(x_min, y_min, x_max, y_max)
        
    return bboxes


def process_poly(segments: list) -> list:
    """
    Convert list of polygons to list of numpy array
    """
    polygons = []
    for poly in segments:
        if poly == []:
            print("DEBUG")
            polygons.append([[0.0, 0.0]]) # Empty polygon?
        polygons.append(np.array(poly, dtype = np.int32))
    
    return polygons


def poly_to_bbox(polygons):
    # Get bounding boxes from a list of polygons
    bboxes = []
    for poly in polygons:
        # TODO: Slow here, fix it
        x_min = min(p[0] for p in poly)
        y_min = min(p[1] for p in poly)
        x_max = max(p[0] for p in poly)
        y_max = max(p[1] for p in poly)
        # Convert list to numpy
        # poly = np.array(poly) # Shape should be (2, N)
        # x_min = min(poly[:, 0])
        # y_min = min(poly[:, 1])
        # x_max = max(poly[:, 0])
        # y_max = max(poly[:, 1])
        
        bboxes.append((x_min, y_min, x_max, y_max))

    return bboxes

def calculate_bbox_area(bbox):
    # BBox should be in the format of (x_min, y_min, x_max, y_max)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    return width * height

def calculate_segment_area(segment, resolution, mode):
    assert mode in ['mask', 'bbox']
    mask = np.zeros(resolution)
    if mode == 'mask':
        polygons = process_poly(segment)
        cv2.fillPoly(mask, polygons, (1, 1, 1))
        area = np.sum(mask)
        
    elif mode == 'bbox':
        bboxes = poly_to_bbox(segment)
        for bbox in bboxes:
            bbox = list(map(round, bbox)) # convert float to int
            cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 1, -1) # -1 mean fill entirely rectangle
        area = np.sum(mask)
        
    return area
        
def segment_iou(segA, segB):
    intersection = np.logical_and(segA, segB)
    union = np.logical_or(segA, segB)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
    
    return iou
