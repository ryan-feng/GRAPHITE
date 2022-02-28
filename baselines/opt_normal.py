import sys
import time

sys.path.append('..')
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from transforms import get_transform_params, get_transformed_images, convert2Network
from utils import run_predictions
from GTSRB.GTSRBNet import GTSRBNet
from GTSRB.GTSRBDataset import GTSRBDataset
from tqdm import tqdm

USE_XFORMS = True
USE_INDEX = True

EVAL_WITH_ALL_XFORMS = False
query_count = 0


def tr_predict(model, image, target, xforms, lbd, theta, index=None):
    global query_count
    query_count += 1
    return model.predict(convert2Network(image.cuda() + lbd * theta.cuda()))


def attack_targeted(model, train_dataset, x0, y0, target, mask, theta_initializer=None, lbd_initializer=None, alpha=5, beta=0.001,
                    iterations=1000):
    """ Attack the original image and return adversarial example of target t
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        t: target
    """
    global query_count
    # STEP I: find initial direction (theta, g_theta)
    best_theta, g_theta = None, float('inf')
    query_count = 0

    timestart = time.time()
    for i, (xi, yi) in enumerate(train_dataset):
        if theta_initializer is not None:
            theta = theta_initializer * mask
            initial_lbd = lbd_initializer * torch.norm(theta)
            theta = theta / torch.norm(theta)
        else:
            theta = (xi - x0) * mask
            initial_lbd = torch.norm(theta)
            theta = theta / torch.norm(theta)
        lbd = fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd, g_theta)
        if lbd < g_theta:
            best_theta, g_theta = theta, lbd
    timeend = time.time()
    time_init = timeend - timestart
    if best_theta is None:
        return torch.zeros(x0.size()), torch.zeros(x0.size()), query_count, 0, torch.zeros(x0.size()), -1
    init_theta, init_g = best_theta.clone(), g_theta

    # STEP II: seach for optimal
    timestart = time.time()
    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta
    for i in tqdm(range(iterations)):
        gradient = torch.zeros(theta.size())
        q = 10
        valid_count = 0
        min_g1 = float('inf')
        for _ in range(q):
            u = torch.randn(theta.size()).type(torch.FloatTensor) * mask
            u = u / torch.norm(u)
            ttt = theta + beta * u
            ttt = ttt / torch.norm(ttt)
            g1 = fine_grained_binary_search_local_targeted(model, x0, y0, target, ttt, initial_lbd=g2,
                                                                  tol=beta / 500)
            if g1 > 100000:
                continue 
            gradient += (g1 - g2) / beta * u
            valid_count += 1
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        if valid_count == 0:
            return x0 + g_theta * best_theta, best_theta, query_count, time_init + time.time() - timestart, None, g_theta
        gradient = 1.0 / valid_count * gradient

        if (i + 1) % 50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (
                i + 1, g1, g2, torch.norm(g2 * theta), query_count))

        min_theta = theta
        min_g2 = g2

        for _ in range(15):
            new_theta = theta - alpha * gradient
            new_theta = new_theta / torch.norm(new_theta)
            new_g2 = fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta,
                                                                      initial_lbd=min_g2, tol=beta / 500)
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta
                min_g2 = new_g2
            else:
                break

        if min_g2 >= g2:
            for _ in range(15):
                alpha = alpha * 0.25
                new_theta = theta - alpha * gradient
                new_theta = new_theta / torch.norm(new_theta)
                new_g2 = fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta,
                                                                          initial_lbd=min_g2, tol=beta / 500)
                if new_g2 < g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    break

        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1

        if g2 < g_theta:
            best_theta, g_theta = theta.clone(), g2

        if alpha < 1e-4:
            alpha = 1.0
            print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
            beta = beta * 0.1
            if (beta < 0.0005):
                break

    print("end", target)
    pred_target = tr_predict(model, x0, target, None, g_theta, best_theta)
    if pred_target == -1:
        g_theta = init_g
        best_theta = init_theta.clone()
    pred_target = tr_predict(model, x0, target, None, g_theta, best_theta)
    print("end2", pred_target)
    timeend = time.time()
    print("\nOpt attack completed: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (
        g_theta, pred_target, query_count, timeend - timestart))
    return x0 + g_theta * best_theta, best_theta, query_count, time_init + timeend - timestart, gradient, g_theta


def fine_grained_binary_search_local_targeted(model, x0, y0, t, theta, initial_lbd=1.0, tol=1e-5, xforms=None,
                                              index=None):
    lbd = initial_lbd
    if tr_predict(model, x0, t, xforms, lbd, theta, index) != t:
        lbd_lo = lbd
        lbd_hi = lbd * 1.01
        while tr_predict(model, x0, t, xforms, lbd_hi, theta, index) != t:
            lbd_hi = lbd_hi * 1.01
            if lbd_hi > 100:
                return float('inf')
    else:
        lbd_hi = lbd
        lbd_lo = lbd * 0.99
        while tr_predict(model, x0, t, xforms, lbd_lo, theta, index) == t:
            lbd_lo = lbd_lo * 0.99

    prev_lbd_mid = None
    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        if prev_lbd_mid is not None and lbd_mid == prev_lbd_mid:
            # need to get unstuck from numerical rounding issues
            tol = (lbd_hi - lbd_lo) + 0.0001

        prev_lbd_mid = lbd_mid
        if tr_predict(model, x0, t, xforms, lbd_mid, theta, index) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi


def fine_grained_binary_search_targeted(model, x0, y0, t, theta, initial_lbd, current_best, xforms=None):
    if initial_lbd > current_best:
        assert False
        if tr_predict(model, x0, t, xforms, current_best, theta) != t:
            return float('inf'), nquery
        lbd = current_best
    else:
        lbd = initial_lbd

    if tr_predict(model, x0, t, xforms, lbd, theta) != t:
        lbd_lo = lbd
        lbd_hi = lbd * 1.01
        while tr_predict(model, x0, t, xforms, lbd_hi, theta) != t:
            lbd_hi = lbd_hi * 1.01
            if lbd_hi > 100:
                return float('inf')
    else:
        lbd_hi = lbd
        lbd_lo = 0.0

    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        if tr_predict(model, x0, t, xforms, lbd_mid, theta) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid

    return lbd_hi
