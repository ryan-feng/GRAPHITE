# Default mask scoring function. High penalty for going below tr_err, otherwise
# it balances size of the mask vs. trivability

import cv2
import torch
import numpy as np

def score_fn(theta, mask, tr_err, object_size, coord = None, threshold=0.75, lbd=5, *args, **kargs):
    if tr_err > threshold:
        return 10000000

    return lbd * mask.sum() / 3 / (object_size) + tr_err
