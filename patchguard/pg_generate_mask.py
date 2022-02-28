import sys
sys.path.append("..")
import re
import sys
import time
import random
import functools
import importlib
import parsearguments
import numpy as np
import cv2
import torch
from coarse_reduction import get_coarse_reduced_mask
from pg_transforms import get_transform_params, get_transformed_images, convert2Network, add_noise
from utils import run_predictions
import pickle
from torchvision.utils import save_image

scoremod = None


def generate_mask(model, img_v_small, img_v, lbl_v, mask, img_t_small, img_t, lbl_t, pt_file, scorefile, HEATMAP_MODE,
                  COARSE_ACCEPT_THRESHOLD, FINE_REJECT_THRESHOLD,
                  num_xforms=100, net_size=32, model_type='CIFAR', patch_size=4, stride_factor=4, 
                  heatmap_file=None, heatmap_out_path=None, max_mask_size=-1):
    """ Attack the original image and return adversarial example
        model: (pytorch model)
        (x0, y0): original image
        mask must be all 0's and 1's and of the resolution that we want to generate noise in
    """

    # Read in arguments
    global scoremod

    # Read in arguments continued
    modulename = scorefile.strip('.py')
    importlib.invalidate_caches()
    scoremod = importlib.import_module(modulename)
    args = parsearguments.getarguments()

    print("Calculating Heatmap...")
    ########################### Stage 1: HEATMAP: Collect all the valid patches and order by computed heatmap ################################
    xforms = get_transform_params(num_xforms, model_type, baseline=True)
    query_count = 0
    timestart = time.time()
    not_mask = torch.ones(mask.size()) - mask
    object_size = mask.sum() / 3

    patch = torch.ones((3, patch_size, patch_size))
    patches = []
    indices = []

    if heatmap_file is None:
        # collect all valid patches
        for i in range(0, mask.size()[1] - patch_size, patch_size // stride_factor):  # 4
            for j in range(0, mask.size()[2] - patch_size, patch_size // stride_factor):  # 4
                new_mask = torch.zeros(mask.size())
                new_mask[:, i:min(i + patch_size, mask.size()[1]), j:min(j + patch_size, mask.size()[2])] = patch
                new_mask = new_mask * mask
                if torch.sum(new_mask) > 0:
                    patches.append(new_mask)
                    indices.append((i, j))
        print("num patches: ", len(patches))

        # compute heatmap and order
        tr_scores, heatmap_query_ct = survey_heatmap(mask, patches, indices, img_v, img_t, img_v_small, img_t_small,
                                                       lbl_t, lbl_v, model, xforms, pt_file, net_size, HEATMAP_MODE,
                                                       plot=True)
        query_count += heatmap_query_ct

        tr_scores_np = np.asarray(tr_scores)
        order = tr_scores_np.argsort()
        patches = [patches[ind] for ind in order]
        indices = [indices[ind] for ind in order]

        if heatmap_out_path is not None:
            with open(heatmap_out_path, 'wb') as f:
                pickle.dump({
                    'heatmap_query_ct': heatmap_query_ct,
                    'patches': patches,
                    'indices': indices,
                    'order': order,
                }, f)

    else:
        with open(heatmap_file, 'rb') as f:
            heatmap = pickle.load(f)
            heatmap_query_ct = heatmap['heatmap_query_ct']
            patches = heatmap['patches']
            indices = heatmap['indices']
            order = heatmap['order']
        query_count += heatmap_query_ct
        patches = patches
        indices = indices
        order = order

    print("Heatmap completed.")

    ########################### Stage 2: COARSE REDUCTION: coarsely remove patches until the high surivability threshold is reached ################################
    args = parsearguments.getarguments();
    coarse_red_mode = args.coarse_mode  # binary or linear
    if coarse_red_mode != 'none':
        direction = "forward"
        print("Coarse reduction start mask using", coarse_red_mode, direction)

        best_score, best_tr, best_mask, coarse_reduction_query_ct, pivot, patches, indices = \
            get_coarse_reduced_mask(mask, object_size, patches, indices, img_v, img_t, lbl_v, lbl_t, model, xforms, pt_file,
                                    net_size,
                                    num_xforms=num_xforms, patch_size=patch_size, err_threshold=COARSE_ACCEPT_THRESHOLD,
                                    coarse_red_mode=coarse_red_mode, direction=direction, args=args)
        coarse_red_nbits = best_mask.sum() / 3
        query_count += coarse_reduction_query_ct

    else:
        best_mask = mask
        coarse_red_nbits = best_mask.sum() / 3

    ########################### Stage 3: FINE REDUCTION: iterate over patches and greedily remove if the mask score improves ################################
    print("Starting fine grained reduction")
    if max_mask_size > 0:
        lbd = 5
        while best_mask.sum() / 3 > max_mask_size:
            best_score, best_tr, best_mask, reduction_query_ct = get_fine_reduced_mask(best_mask, object_size, patches,
                                                                                    indices, img_v, img_t_small, lbl_v,
                                                                                    lbl_t, model, xforms, pt_file,
                                                                                    net_size,
                                                                                    err_threshold=FINE_REJECT_THRESHOLD,
                                                                                    HEATMAP_MODE=HEATMAP_MODE,
                                                                                    scoremod=scoremod, lbd=lbd,
                                                                                    max_mask_size=max_mask_size)
            lbd += 2.5
    else:
        best_score, best_tr, best_mask, reduction_query_ct = get_fine_reduced_mask(best_mask, object_size, patches,
                                                                                indices, img_v, img_t_small, lbl_v,
                                                                                lbl_t, model, xforms, pt_file, net_size,
                                                                                err_threshold=FINE_REJECT_THRESHOLD,
                                                                                HEATMAP_MODE=HEATMAP_MODE,
                                                                                scoremod=scoremod)
    reducer_nbits = best_mask.sum() / 3
    query_count += reduction_query_ct

    ############### Print results, save out and returned initialization point found ##############3
    print('-' * 32)
    print("End of Mask Generation summary:")
    print("queries", reduction_query_ct)
    print("final (coarse_red to reducer) bits", coarse_red_nbits.item(), reducer_nbits.item())
    print("final tr", best_tr)
    print("final area ratio", (reducer_nbits / object_size).item())
    print("queries used", query_count)
    print('-' * 32)

    best_theta = (img_t_small - img_v_small) * best_mask
    attacked = add_noise(img_v, best_mask, 1.0, best_theta)

    return attacked, query_count, best_mask, reducer_nbits.item(), best_tr


def get_heatmap(start_mask, patches, indices, img_v, img_t, lbl_v, lbl_t, model, xforms, pt_file, net_size,
                init_tr=0.5, plot=False):
    object_size = start_mask.sum() / 3

    # setup images
    img_v_np = img_v.permute(1, 2, 0).numpy()
    img_v_np = cv2.resize(img_v_np, (start_mask.size()[2], start_mask.size()[1]))
    img_v_small = torch.from_numpy(img_v_np).permute(2, 0, 1)

    if img_t.size()[1] != start_mask.size()[1] or img_t.size()[2] != start_mask.size()[2]:
        img_t_np = img_t.permute(1, 2, 0).numpy()
        img_t_np = cv2.resize(img_t_np, (start_mask.size()[2], start_mask.size()[1]))
        img_t = torch.from_numpy(img_t_np).permute(2, 0, 1)

    heatmap_query_ct = 0

    tr_scores = []

    # iterate over patches and compute transform_robustness without each individual patch
    for i in range(len(patches)):
        patch = patches[i]
        next_mask = start_mask * (torch.ones(start_mask.size()) - patch)
        theta = (img_t - img_v_small) * next_mask
        xform_imgs = get_transformed_images(img_v, next_mask, xforms, 1.0, theta, pt_file,
                                            net_size=net_size)
        success_rate, query_ct = run_predictions(model, xform_imgs, lbl_v, lbl_t)
        heatmap_query_ct += query_ct
        tr_scores.append(1 - success_rate)

    return tr_scores, heatmap_query_ct


def get_fine_reduced_mask(start_mask, object_size, patches, indices, img_v, img_t, lbl_v, lbl_t, model, xforms, pt_file,
                     net_size, err_threshold=0.75, HEATMAP_MODE=None, scoremod=None, lbd=5, max_mask_size=-1):
    # STAGE 3: Fine reduction
    reduction_query_ct = 0

    # set up images
    img_v_np = img_v.permute(1, 2, 0).numpy()
    img_v_np = cv2.resize(img_v_np, (start_mask.size()[2], start_mask.size()[1]))
    img_v_small = torch.from_numpy(img_v_np).permute(2, 0, 1)

    if img_t.size()[1] != img_v_small.size()[1] or img_t.size()[2] != img_v_small.size()[2]:
        img_t_np = img_t.permute(1, 2, 0).numpy()
        img_t_np = cv2.resize(img_t_np, (start_mask.size()[2], start_mask.size()[1]))
        img_t = torch.from_numpy(img_t_np).permute(2, 0, 1)
    img_t_small = img_t

    theta = (img_t - img_v_small) * start_mask

    # get initial transform_robustness
    xform_imgs = get_transformed_images(img_v, start_mask, xforms, 1.0, theta, pt_file, net_size=net_size)
    success_rate, query_ct = run_predictions(model, xform_imgs, lbl_v, lbl_t)
    reduction_query_ct += query_ct

    init_tr = (1 - success_rate)
    print("init_tr", init_tr)
    print("init bits", start_mask.sum() / 3)

    best_score = scoremod.score_fn(theta + img_v_small, start_mask, success_rate, object_size, None,
                                   threshold=err_threshold, lbd=lbd)
    best_tr = init_tr
    best_mask = start_mask
    last_heatmap_mask = best_mask

    new_patches, patches_examined, peek_cnt, zero_grad = [], 0, 8, 0
    j = 0
    # iterate over patches, greedily remove if the score improves
    while patches:
        j = j + 1

        # highest tr is now always at end, for pop()
        next_patch, next_indice = patches.pop(), indices.pop()
        if torch.max(next_patch * best_mask) == 0:
            continue
        patches_examined += 1
        next_mask = best_mask * (torch.ones(best_mask.size()) - next_patch)
        theta = (img_t - img_v_small) * next_mask

        xform_imgs = get_transformed_images(img_v, next_mask, xforms, 1.0, theta, pt_file, net_size=net_size)
        success_rate, query_ct = run_predictions(model, xform_imgs, lbl_v, lbl_t)
        reduction_query_ct += query_ct

        score = scoremod.score_fn(theta + img_v_small, next_mask, success_rate, object_size, next_indice,
                                  best_score=best_score, threshold=err_threshold, lbd=lbd)
        if score < best_score:
            best_score = score
            best_mask = next_mask
            best_tr = 1 - success_rate
            nbits = best_mask.sum() / 3
            if max_mask_size > 0 and nbits < max_mask_size:
                break
        else:
            new_patches.append(next_patch)

    patches = new_patches.copy()

    return best_score, best_tr, best_mask, reduction_query_ct


def survey_heatmap(mask, patches, indices, img_v, img_t, img_v_small, img_t_small, lbl_t, lbl_v, model, xforms, pt_file,
                   net_size, HEATMAP_MODE, plot=False):
    ''' encapsulation of the transform_robustness measurements heatmap to compute
           random
           target wrt victim or 
           victim wrt target '''
    if HEATMAP_MODE == 'Random':
        tr_scores = [random.random() for i in range(len(patches))]
        heatmap_query_ct = 0
    elif HEATMAP_MODE == 'Victim':
        tr_scores, heatmap_query_ct = get_heatmap(mask, patches, indices, img_t, img_v_small, lbl_t, lbl_v, model,
                                                    xforms, pt_file, net_size, plot=plot) 
    else:  ## Target mode
        tr_scores, heatmap_query_ct = get_heatmap(mask, patches, indices, img_v, img_t_small, lbl_v, lbl_t, model,
                                                    xforms, pt_file, net_size, plot=plot)

    return tr_scores, heatmap_query_ct
