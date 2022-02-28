import sys
import time
import random
import importlib
import numpy as np
import cv2
import torch 
from transforms import get_transform_params, get_transformed_images, convert2Network
from utils import run_predictions


def get_coarse_reduced_mask(start_mask, object_size, patches, indices, img_v, img_t, lbl_v, lbl_t, model, xforms, pt_file, net_size, num_xforms = 1000, patch_size = 4, err_threshold = 0.5, coarse_red_mode = 'binary', direction = "forward", args = None, debug = False, model_type = 'GTSRB', init_theta = None):
    if args is not None:
        coarse_red_mode = args.coarse_mode  # binary or linear

    coarse_reduction_query_ct = 0

    # Format images
    img_v_np = img_v.permute(1, 2, 0).numpy()
    img_v_np = cv2.resize(img_v_np, (start_mask.size()[2], start_mask.size()[1]))
    img_v_small = torch.from_numpy(img_v_np).permute(2, 0, 1)

    if img_t.size()[1] != img_v_small.size()[1] or img_t.size()[2] != img_v_small.size()[2]:
        img_t_np = img_t.permute(1, 2, 0).numpy()
        img_t_np = cv2.resize(img_t_np, (start_mask.size()[2], start_mask.size()[1]))
        img_t_small = torch.from_numpy(img_t_np).permute(2, 0, 1)
    else:
        img_t_small = img_t

    # get initial transform_robustness
    if init_theta is None:
        theta = (img_t_small - img_v_small) * start_mask
    else:
        theta = init_theta * start_mask
    xform_imgs = get_transformed_images(img_v, start_mask, xforms, 1.0, theta, pt_file, net_size = net_size, model_type = model_type)
    success_rate, query_ct = run_predictions(model, xform_imgs, lbl_v, lbl_t)
    coarse_reduction_query_ct += query_ct

    print("init_tr", 1 - success_rate)
    print("init bits", start_mask.sum() / 3)

    if "binary" not in coarse_red_mode:  
        # linear-based coarse_reduction of the start mask
        pivot, best_mask, best_tr, query_ct = \
                perform_linear_coarse_reduction(patches, err_threshold, lbl_v, lbl_t, img_t_small, img_v_small, img_v, model, 
                                            xforms, start_mask, pt_file, net_size, model_type = model_type, init_theta = init_theta)

    else:  
        # binary-search-based coarse_reduction of the start mask
        pivot, best_mask, best_tr, query_ct = \
                perform_binary_coarse_reduction(patches, err_threshold, lbl_v, lbl_t, img_t_small, img_v_small, img_v, model,
                                            xforms, start_mask, pt_file, net_size, direction, model_type = model_type, init_theta = init_theta)

    patches = patches[:]       # reduce will examine all patches
    indices = indices[:]

    coarse_reduction_query_ct += query_ct

    print("mask coarse_reduction completed...")
    print('-'*32)
    print("COARSE_REDUCTION: mode ", coarse_red_mode)
    print("COARSE_REDUCTION: direction", direction)
    print("COARSE_REDUCTION: num. patches incorporated", pivot)
    print("COARSE_REDUCTION: final bits in mask", ((best_mask.sum() / 3).item()))
    print("COARSE_REDUCTION: final tr of mask", best_tr)
    print("COARSE_REDUCTION: final area ratio", ((best_mask.sum() / 3) / object_size))
    print("COARSE_REDUCTION: queries used", coarse_reduction_query_ct)
    print('-'*32)

    best_score = 9999999 # large value, unused

    return best_score, best_tr, best_mask, coarse_reduction_query_ct, pivot, patches, indices


def perform_linear_coarse_reduction(patches, err_threshold, lbl_v, lbl_t, img_t_small, img_v_small, img_v, model, xforms, start_mask, pt_file, net_size, model_type = 'GTSRB', init_theta = None):
    linstart = time.time()
    best_mask, best_tr = torch.zeros(start_mask.size()), 0
    tot_queries = 0
    for i in range(1): 
        new_patches = []
        for j in range(len(patches)):
            next_patch = patches[j]
            next_mask = best_mask + (torch.zeros(best_mask.size()) + next_patch)
            next_mask = torch.where(next_mask > 0, torch.ones(best_mask.size()), torch.zeros(best_mask.size())) 
            if (next_mask - best_mask).sum() == 0: continue
            if init_theta is None:
                theta = (img_t_small - img_v_small) * next_mask
            else:
                theta = init_theta * next_mask
            xform_imgs = get_transformed_images(img_v, next_mask, xforms, 1.0, theta, pt_file, net_size = net_size, model_type = model_type)
            success_rate, query_ct = run_predictions(model, xform_imgs, lbl_v, lbl_t)
            tot_queries += query_ct
            best_mask = next_mask
            best_tr = 1 - success_rate
            nbits = (best_mask.sum() / 3)
            print("Coarse Reduction: tr: %.2f" % (1 - success_rate), "bits: %d" % nbits.item(), 'time', time.time() - linstart)
            if best_tr >= (1 - err_threshold):
                print("COARSE_REDUCTION: TR GOAL REACHED", (1 - success_rate))
                break
            else:
                new_patches.append(next_patch)
        patches = new_patches.copy()
        pivot = j
    return pivot, best_mask, best_tr, tot_queries


def perform_binary_coarse_reduction(patches, err_threshold, lbl_v, lbl_t, img_t_small, img_v_small, img_v, model,
                                xforms, start_mask, pt_file, net_size, direction="forward", model_type = 'GTSRB', init_theta = None):
    tot_queries = 0
    patch_cache = {}
    binsearch_start = time.time()

    def evaluate_transform_robustness_at_pivot(pivot, get_mask=False):
        """ binary search plug-in that evaluates whether functional condition is met at specified pivot """
        nonlocal tot_queries
        best_mask = get_accumulated_mask_up_to_pivot(pivot, start_mask, patches, patch_cache)

        if init_theta is None:
            theta = (img_t_small - img_v_small) * best_mask
        else:
            theta = init_theta * best_mask
        xform_imgs = get_transformed_images(img_v, best_mask, xforms, 1.0, theta, pt_file, net_size = net_size, model_type = model_type)
        success_rate, query_ct = run_predictions(model, xform_imgs, lbl_v, lbl_t)

        tot_queries += query_ct
        return (1- success_rate, best_mask) if get_mask else 1 - success_rate

    def get_accumulated_mask_up_to_pivot(pivot, start_mask, patches, patch_cache={}):
        best_mask = torch.zeros(start_mask.size())
        ordering = patches[:pivot] 
        for next_patch in ordering:
            next_mask = best_mask + (torch.zeros(best_mask.size()) + next_patch)
            next_mask = torch.where(next_mask > 0, torch.ones(best_mask.size()), torch.zeros(best_mask.size()))
            best_mask = next_mask
        patch_cache[pivot] = best_mask.clone()
        return best_mask.clone()

    # binary search leftmost pivot value for which tr exceeeds specificed threshold if one exists
    nums = list(range(len(patches)))
    pivot = -1
    threshold = 1 - err_threshold
    n = len(nums)
    mi = -1
    if n == 1: mi = 0

    if mi < 0:
        lo, hi = 0, n-1
        while lo <= hi:
            mi = lo + (hi-lo)//2
            score = evaluate_transform_robustness_at_pivot(mi)
            if score >= threshold:
                if mi > 0:
                    lo, mi, hi = lo, mi, mi-1
                    continue
                else:
                    break
            else:  # score < threshold:
                lo, mi, hi = mi+1, mi+1, hi

    pivot = mi
    assert 0 <= pivot <= n

    best_tr, best_mask = evaluate_transform_robustness_at_pivot(pivot, get_mask=True)

    return pivot, best_mask, best_tr, tot_queries
