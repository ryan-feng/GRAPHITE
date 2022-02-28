import cv2
import torch
import numpy as np

def score_fn(theta, mask, tr_err, object_size, coord = None, threshold=0.75, *args, **kargs):
    if tr_err > threshold:
        return 10000000

    return 25 * mask.sum() / 3 / (object_size) + tr_err
