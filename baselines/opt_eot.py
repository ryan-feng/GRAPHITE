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
from transforms import get_transform_params, get_transformed_images, convert2Network, add_noise
from utils import run_predictions
from GTSRB.GTSRBNet import GTSRBNet
from GTSRB.GTSRBDataset import GTSRBDataset
from tqdm import tqdm

USE_XFORMS = True
USE_INDEX = True

EVAL_WITH_ALL_XFORMS = False
query_count = 0


def tr_predict(model, image, mask, target, xforms, xforms_pt_file, lbd, theta, index=None):
    global query_count
    if index is not None:
        xforms = [xforms[index]]
    try:
        xform_imgs = get_transformed_images(image, mask, xforms, lbd.item(), theta, xforms_pt_file)
    except:
        xform_imgs = get_transformed_images(image, mask, xforms, lbd, theta, xforms_pt_file)
    success_rate, query_ct = run_predictions(model, xform_imgs, 1, target)
    query_count += len(xforms)
    if success_rate <= 0.2:
        return target
    else:
        return -1


def attack_targeted(model, train_dataset, x0, y0, target, mask, xforms, xforms_pt_file, theta_initializer=None, lbd_initializer=None, alpha=5, beta=0.001,
                    iterations=1000):
    """ Attack the original image and return adversarial example of target t
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        t: target
    """
    global query_count

    if tr_predict(model, x0, mask, target, xforms, xforms_pt_file, 1.0, torch.zeros((x0.size()))) == target:
        print("Image classified as target. No need to attack.")
        return x0, 0, 0

    # STEP I: find initial direction (theta, g_theta)

    num_samples = 100
    best_theta, g_theta = None, float('inf')
    query_count = 0


    timestart = time.time()
    for i, (xi, yi) in enumerate(train_dataset):
        result = xi
        if theta_initializer is not None:
            theta = theta_initializer * mask
            initial_lbd = lbd_initializer * torch.norm(theta)
            theta = theta / torch.norm(theta)
        else:
            theta = (xi - x0) * mask
            initial_lbd = torch.norm(theta)
            theta = theta / torch.norm(theta)
        lbd = fine_grained_binary_search_targeted(model, x0, mask, y0, target, theta, initial_lbd, g_theta,
                                                         xforms=xforms, xforms_pt_file=xforms_pt_file)
        if lbd < g_theta:
            best_theta, g_theta = theta, lbd
            init_theta, init_g = theta.clone(), lbd.clone()
    timeend = time.time()
    time_init = timeend - timestart
    if best_theta is None:
        return torch.zeros(x0.size()), torch.zeros(x0.size()), query_count, 0, torch.zeros(x0.size()), -1

    # STEP II: seach for optimal
    timestart = time.time()
    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta
    for i in tqdm(range(iterations)):
        if USE_INDEX:
            avg_gradient = torch.zeros(theta.size())
            for j in tqdm(range(len(xforms))):
                gradient = torch.zeros(theta.size())
                q = 10
                valid_count = 0
                min_g1 = float('inf')
                for _ in range(q):
                    u = torch.randn(theta.size()).type(torch.FloatTensor) * mask
                    u = u / torch.norm(u)
                    ttt = theta + beta * u
                    ttt = ttt / torch.norm(ttt)
                    g1 = fine_grained_binary_search_local_targeted(model, x0, mask, y0, target, ttt, initial_lbd=g2,
                                                                          tol=beta, xforms=xforms, xforms_pt_file=xforms_pt_file, index=j)
                    if g1 > 100000:
                        continue
                    valid_count += 1
                    gradient += (g1 - g2) / beta * u
                    if g1 < min_g1:
                        min_g1 = g1
                        min_ttt = ttt
                if valid_count == 0:
                    return x0 + g_theta * best_theta, best_theta, query_count, time_init + time.time() - timestart, None, g_theta
                gradient = 1.0 / valid_count * gradient
                avg_gradient += gradient
            avg_gradient /= len(xforms)
            gradient = avg_gradient
        else:
            gradient = torch.zeros(theta.size())
            q = 10
            min_g1 = float('inf')
            for _ in range(q):
                u = torch.randn(theta.size()).type(torch.FloatTensor) * mask
                u = u / torch.norm(u)
                ttt = theta + beta * u
                ttt = ttt / torch.norm(ttt)
                g1 = fine_grained_binary_search_local_targeted(model, x0, mask, y0, target, ttt, initial_lbd=g2,
                                                                      tol=beta, xforms=xforms, xforms_pt_file=xforms_pt_file)
                if g1 > 100000:
                    continue
                valid_count += 1
                gradient += (g1 - g2) / beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0 / valid_count * gradient

        if (i + 1) % 50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (
                i + 1, g1, g2, torch.norm(g2 * theta), query_count))

        min_theta = theta
        min_g2 = g2

        for _ in range(15):
            new_theta = theta - alpha * gradient
            new_theta = new_theta / torch.norm(new_theta)
            new_g2 = fine_grained_binary_search_local_targeted(model, mask, x0, y0, target, new_theta,
                                                                      initial_lbd=min_g2, tol=beta, xforms=xforms, xforms_pt_file=xforms_pt_file)
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
                new_g2 = fine_grained_binary_search_local_targeted(model, mask, x0, y0, target, new_theta,
                                                                          initial_lbd=min_g2, tol=beta,
                                                                          xforms=xforms, xforms_pt_file=xforms_pt_file)
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
    pred_target = tr_predict(model, x0, mask, target, xforms, xforms_pt_file, g_theta, best_theta)
    if pred_target == -1:
        g_theta = init_g
        best_theta = init_theta.clone()
    pred_target = tr_predict(model, x0, mask, target, xforms, xforms_pt_file, g_theta, best_theta)
    print("end2", pred_target)
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (
        g_theta, pred_target, query_count, timeend - timestart))
    return x0 + g_theta * best_theta, best_theta, query_count, time_init + timeend - timestart, gradient, g_theta


def fine_grained_binary_search_local_targeted(model, x0, mask, y0, t, theta, initial_lbd=1.0, tol=1e-3, xforms=None, xforms_pt_file=None,
                                              index=None):
    lbd = initial_lbd

    if tr_predict(model, x0, mask, t, xforms, xforms_pt_file, 0, theta, index) == t:
        return 0

    if tr_predict(model, x0, mask, t, xforms, xforms_pt_file, lbd, theta, index) != t:
        lbd_lo = lbd
        if lbd == 0: ## just in case to avoid infinite loop
            lbd = 1.0
        lbd_hi = lbd * 1.10
        while tr_predict(model, x0, mask, t, xforms, xforms_pt_file, lbd_hi, theta, index) != t:
            lbd_hi = lbd_hi * 1.10
            if lbd_hi > 100:
                return float('inf')
    else:
        lbd_hi = lbd
        lbd_lo = lbd * 0.90
        while tr_predict(model, x0, mask, t, xforms, xforms_pt_file, lbd_lo, theta, index) == t:
            lbd_lo = lbd_lo * 0.90

    prev_lbd_mid = None
    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        if prev_lbd_mid is not None and lbd_mid == prev_lbd_mid:
            # need to get unstuck from numerical rounding issues
            tol = (lbd_hi - lbd_lo) + 0.0001

        prev_lbd_mid = lbd_mid
        if tr_predict(model, x0, mask, t, xforms, xforms_pt_file, lbd_mid, theta, index) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi


def fine_grained_binary_search_targeted(model, x0, mask, y0, t, theta, initial_lbd, current_best, xforms=None, xforms_pt_file=None):
    if initial_lbd > current_best:
        assert False
        if tr_predict(model, x0, mask, t, xforms, xforms_pt_file, current_best, theta) != t:
            return float('inf')
        lbd = current_best
    else:
        lbd = initial_lbd

    if tr_predict(model, x0, mask, t, xforms, xforms_pt_file, lbd, theta) != t:
        lbd_lo = lbd
        if lbd == 0: ## just in case to avoid infinite loop
            lbd = 1.0
        lbd_hi = lbd * 1.10
        while tr_predict(model, x0, mask, t, xforms, xforms_pt_file, lbd_hi, theta) != t:
            lbd_hi = lbd_hi * 1.10
            if lbd_hi > 100:
                return float('inf')
    else:
        lbd_hi = lbd
        lbd_lo = 0.0

    while (lbd_hi - lbd_lo) > 1e-3:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        if tr_predict(model, x0, mask, t, xforms, xforms_pt_file, lbd_mid, theta) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid

    return lbd_hi
