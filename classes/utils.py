import logging
import os
import random
import cv2
import pickle
import numpy as np

from itertools import product

cv2.setNumThreads(0)

def load_pickled_dict(path):
    assert isinstance(path, str)
    if not os.path.exists(path):
        raise f'Path [{path}] does not exist'
    with open(path, 'rb') as f:
        return pickle.load(f)

def read_text_stat(roi, chars_patterns=None):
    '''
    Read pixel-text: 
    1) split by countours
    2) match with patterns
    '''
    if not chars_patterns:
        return
    if not isinstance(chars_patterns, dict):
        return
    _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    result = ''
    for contour, (k, v) in product(reversed(contours), chars_patterns.items()):
        x, y, w, h = cv2.boundingRect(contour)
        c = get_roi(roi, x, y, h, w)
        if np.array_equal(v, c):
            result += str(k)[:1]
            continue
    return int(result or 0)

def get_roi(img, x1:int, y1:int, w:int, h:int):
    if img is None:
        return
    return img[y1:y1+h, x1:x1+w]

def get_roi3d(img, x1:int, y1:int, w:int, h:int):
    if img is None:
        return
    return img[y1:y1+h, x1:x1+w,:]

def resize_img(img, scale_factor=0.5):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    return cv2.resize(img, (width, height))