import time
import random
import numpy as np
import cv2
import torch 
from transforms import get_transform_params, get_transformed_images, convert2Network, add_noise
from utils import run_predictions

def boost(model, x0_large, y0, mask, target_example, target = None, beta = 1, iterations = 30, pt_file = 'images/Points/14.csv', num_xforms = 1000, net_size = 32, model_type = 'GTSRB', goal = None, eta = 500, bt = False, budget_factor = 200, init_theta = None, square_mask = False, early_boost_exit = False):
    """ Input requirements:
        x0_large: must be a torch tensor and must be the output size of the final desired result (eg, the larger size)
        y0: the victim label of x0_large
        mask: Must consist of ONLY 0's and 1's. If any resizing is done before calling attack(), you must threshold it out so it's only 0's and 1's
              Must be a torch tensor and must be the size of the noise to be generated (eg, 32 x 32)
        target_example: The example target image. Must be a torch tensor and must be the size of the noise to be generated (eg, 32 x 32)
        target: the desired target label

        Attack the original image and return adversarial example
        model: (pytorch model)
        (x0, y0): original image
        mask must be all 0's and 1's and of the resolution that we want to generate noise in
    """
    xforms = get_transform_params(num_xforms, model_type)

    query_count = 0

    timestart = time.time()

    ######### Initialize ##############
    theta = None

    x0_lg_np = x0_large.permute(1, 2, 0).numpy()
    x0_np = cv2.resize(x0_lg_np, (mask.size()[2], mask.size()[1]))
    x0 = torch.from_numpy(x0_np).permute(2, 0, 1)

    if init_theta is not None:
        theta = init_theta * mask
    else:
        theta = (target_example - x0) * mask
    theta_np = theta.permute(1, 2, 0).numpy()

    theta_np_large = cv2.resize(theta_np, (x0_large.size()[2], x0_large.size()[1]))
    theta_np_large_torch = torch.from_numpy(theta_np_large).permute(2, 0, 1)
    comb_large = x0_large + theta_np_large_torch
    comb_np = comb_large.permute(1, 2, 0).numpy()
    comb_np = cv2.resize(comb_np, (mask.size()[2], mask.size()[1]))
    comb_torch = torch.from_numpy(comb_np).permute(2, 0, 1)

    xform_imgs = get_transformed_images(x0_large, mask, xforms, 1.0, theta, pt_file, net_size = net_size, model_type = model_type)
    success_rate, query_ct = run_predictions(model, xform_imgs, y0, target)
    query_count += query_ct

    best_theta, best_eps = theta, success_rate

    timeend = time.time()
    print("==========> Found success rate %.4f in %.4f seconds using %d queries" % (best_eps, timeend-timestart, query_count))

    if x0_large is not None:
        init_theta_np = best_theta.permute(1, 2, 0).numpy()
        init_theta_np = cv2.resize(init_theta_np, (x0_large.size()[2], x0_large.size()[1]))
        init_theta_lg_torch = torch.from_numpy(init_theta_np).permute(2, 0, 1)
        init_attacked = x0_large + init_theta_lg_torch

        init_np = init_attacked.permute(1, 2, 0).numpy()
        init_np = cv2.cvtColor(init_np, cv2.COLOR_RGB2BGR)

    ######### End Initialize ##############
    
    timestart = time.time()

    theta, eps = best_theta.clone(), best_eps

    opt_count = 0

    ####### gradient free optimization steps #######
    iters_taken = 0
    trivial_iters = 0
    for i in range(iterations):
        iters_taken += 1
        print('iter: ', i)
        gradient = torch.zeros(theta.size())
        q = 10 
        min_g1 = float('inf')

        # Take q samples of random Gaussian noise to use as new directions, calculate transform_robustness
        # Used for gradient estimate
        for _ in range(q):
            u = torch.randn(theta.size()).type(torch.FloatTensor) * mask
            u = u/torch.norm(u)
            ttt = theta+beta * u

            xform_imgs = get_transformed_images(x0_large, mask, xforms, 1.0, ttt, pt_file, net_size = net_size, model_type = model_type)
            eps_ttt, opt_ct = run_predictions(model, xform_imgs, y0, target)
            opt_count += opt_ct

            gradient += (eps_ttt - eps)/beta * u
            print("new eps eps beta", eps_ttt, eps, beta)

            if bt and eps_ttt < min_g1:
                min_g1 = eps_ttt
                min_ttt = ttt

        gradient = 1.0/q * gradient

        new_eps = 1.0
        new_theta = None

        if bt:
            min_theta = theta
            min_new_eps = eps

            alpha = 500

            for _ in range(5):
                new_theta = theta - alpha * gradient
                xform_imgs = get_transformed_images(x0_large, mask, xforms, 1.0, new_theta, pt_file, net_size = net_size, model_type = model_type)
                new_eps, opt_ct = run_predictions(model, xform_imgs, y0, target)
                opt_count += opt_ct
                alpha = alpha * 2
                if new_eps < min_new_eps:
                    min_theta = new_theta
                    min_new_eps = new_eps
                else:
                    break

            if min_new_eps >= eps:
                for _ in range(5):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    xform_imgs = get_transformed_images(x0_large, mask, xforms, 1.0, new_theta, pt_file, net_size = net_size, model_type = model_type)
                    new_eps, opt_ct = run_predictions(model, xform_imgs, y0, target)
                    opt_count += opt_ct
                    if new_eps < eps:
                        min_theta = new_theta
                        min_new_eps = new_eps
                        break

            if min_new_eps == eps:
                min_theta = theta - eta * gradient
                xform_imgs = get_transformed_images(x0_large, mask, xforms, 1.0, min_theta, pt_file, net_size = net_size, model_type = model_type)
                min_new_eps, opt_ct = run_predictions(model, xform_imgs, y0, target)
                opt_count += opt_ct

            if min_new_eps <= min_g1:
                new_theta, new_eps = min_theta, min_new_eps
            else:
                new_theta, new_eps = min_ttt, min_g1

        else:
            # Take gradient step
            new_theta = theta - eta * gradient
            xform_imgs = get_transformed_images(x0_large, mask, xforms, 1.0, new_theta, pt_file, net_size = net_size, model_type = model_type)
            new_eps, opt_ct = run_predictions(model, xform_imgs, y0, target)
            opt_count += opt_ct

    
        if new_eps < best_eps:
            best_theta, best_eps = new_theta.clone(), new_eps

        theta, eps = new_theta.clone(), new_eps

        
        if (opt_count + query_count + num_xforms * 11) > (budget_factor * num_xforms):
            break

        if square_mask and i == 0 and best_eps >= 0.95 and early_boost_exit:
            break

        best_theta_inter_full_lg = best_theta.permute(1, 2, 0).numpy()
        best_theta_inter_full_lg = cv2.resize(best_theta_inter_full_lg, (x0_large.size()[2], x0_large.size()[1]))
        best_theta_inter_torch = torch.from_numpy(best_theta_inter_full_lg).permute(2, 0, 1)
        intermediate2 = x0_large + 1.0 * best_theta_inter_torch

        intermediate2_np = intermediate2.permute(1, 2, 0).numpy()
        intermediate2_np = cv2.cvtColor(intermediate2_np, cv2.COLOR_RGB2BGR)

    best_theta_np = best_theta.permute(1, 2, 0).numpy()
    best_theta_np_lg = cv2.resize(best_theta_np, (x0_large.size()[2], x0_large.size()[1]))
    best_theta_lg_torch = torch.from_numpy(best_theta_np_lg).permute(2, 0, 1)
    if model_type == 'GTSRB':
        adv_example = add_noise(x0_large, mask, 1.0, best_theta)
    else:
        adv_example = add_noise(x0_large, mask, 1.0, best_theta, clip = False)

    x0_lg_numpy = x0_large.permute(1, 2, 0).numpy()
    mask_np = mask.permute(1, 2, 0).numpy()
    mask_np = cv2.resize(mask_np, (x0_large.size()[2], x0_large.size()[1]))
    mask_large = torch.from_numpy(mask_np).permute(2, 0, 1)
    perturb = torch.where(mask_large > 0.0, adv_example, torch.ones(adv_example.size()))

    target = model.predict(convert2Network(adv_example, net_size = net_size, model_type = model_type))
    timeend = time.time()
    print("final iters_taken: ", iters_taken)

    return adv_example, (1 - best_eps), perturb, query_count + opt_count, best_theta, torch.clamp(x0 + best_theta, 0, 1)
