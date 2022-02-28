import time
import random 
import numpy as np
import cv2
import sys
from utils import run_predictions
import torch 
import torchvision.transforms as transforms
from torchvision import models
from GTSRB.GTSRBDataset import GTSRBDataset
from GTSRB.GTSRBNet import GTSRBNet
from OpenALPR.OpenALPRBorderNet import OpenALPRBorderNet
from boost import boost
from transforms import get_transform_params, get_transformed_images, convert2Network
from generate_mask import generate_mask
from shutil import copyfile
import parsearguments

def attack_network(model, img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta= 1, num_xforms_mask = 100, num_xforms_boost = 1000, net_size = 32, noise_size = 32, model_type = 'GTSRB', mask_outer=None, mask_inner=None):

    eta = 500
    import getpass
    username = getpass.getuser()
    args = parsearguments.getarguments()
    coarse_mode = args.coarse_mode
    tag = args.tag
    bt = args.bt
    

    ######################### PROCESS AND READ INPUTS #########################################
    out_str_base = username + "_" + str(lbl_v) + '_' + str(lbl_t) + '_' + str(reduceerror)  + '_' + str(coarseerror) + '_' +  str(coarse_mode) + '_' + scorefile + "_" ;
    output_base = args.out_path
    if model_type != 'GTSRB':
        output_base += model_type + '/'
    img = cv2.imread(img_v)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32) / 255.0

    ### Adjust noise size in case of non-square images
    if noise_size == -1:
        noise_size = min(img.shape[0], img.shape[1])
    if img.shape[0] > img.shape[1]: # tall and narrow
        noise_x = noise_size
        noise_y = int(round(noise_size / img.shape[1] * img.shape[0]))
    else: # wide and short or square 
        noise_y = noise_size 
        noise_x = int(round(noise_size / img.shape[0] * img.shape[1]))
    img_small = cv2.resize(img, (noise_x, noise_y))
    img_torch = torch.from_numpy(img).permute(2, 0, 1)
    img_small_torch = torch.from_numpy(img_small).permute(2, 0, 1)

    mask_file = None
    if mask is not None and model_type != 'OpenALPRBorder':
        mask_file = mask
        mask_full = cv2.imread(mask)
        if mask_full.shape[0] != noise_y or mask_full.shape[1] != noise_x:
            mask = cv2.resize(mask_full, (noise_x, noise_y))

        else:
            mask = mask_full

        mask = np.where(mask > 128, 255, 0) 
        mask = torch.from_numpy(mask).permute(2, 0, 1) / 255.0

    if img_t is not None:
        tar_large = cv2.imread(img_t)
        tar = cv2.resize(tar_large, (noise_x, noise_y))
        tar = cv2.cvtColor(tar, cv2.COLOR_RGB2BGR)
        tar = tar / 255.0
        tar = torch.from_numpy(tar).permute(2, 0, 1).float()
        tar_large = cv2.cvtColor(tar_large, cv2.COLOR_RGB2BGR)
        tar_large = tar_large / 255.0
        tar_large = torch.from_numpy(tar_large).permute(2, 0, 1).float()

    if mask_outer is not None:
        mask_outer_file = mask_outer
        mask_outer_full = cv2.imread(mask_outer)
        if mask_outer_full.shape[0] != noise_y or mask_outer_full.shape[1] != noise_x:
            mask_outer = cv2.resize(mask_outer_full, (noise_x, noise_y))

        else:
            mask_outer = mask_outer_full

        mask_outer = np.where(mask_outer > 128, 255, 0) 
        mask_outer = torch.from_numpy(mask_outer).permute(2, 0, 1) / 255.0

    if mask_inner is not None:
        mask_inner_file = mask_inner
        mask_inner_full = cv2.imread(mask_inner)
        if mask_inner_full.shape[0] != noise_y or mask_inner_full.shape[1] != noise_x:
            mask_inner = cv2.resize(mask_inner_full, (noise_x, noise_y))

        else:
            mask_inner = mask_inner_full

        mask_inner = np.where(mask_inner > 128, 255, 0) 
        mask_inner = torch.from_numpy(mask_inner).permute(2, 0, 1) / 255.0



    ##################################### STAGE 1: Generate mask ##################################
    nbits = 1000000
    total_query_count = 0
    next_theta = None
    for i in range(3):
        tmp = None
        tmp_small = None
        if model_type == 'OpenALPRBorder':
            tmp = img_torch.clone()
            tmp_small = img_small_torch.clone()
            img_small_torch = torch.where(mask_outer > 0.5, tar, img_small_torch)
            mask_outer_large = mask_outer.permute(1, 2, 0).numpy()
            mask_outer_large = cv2.resize(mask_outer_large, (img_torch.size()[2], img_torch.size()[1]))
            mask_outer_large = torch.from_numpy(mask_outer_large).permute(2, 0, 1)
            img_torch = torch.where(mask_outer_large > 0.5, tar_large, img_torch)
            mask = mask_inner

        if i < 1:
            patch_size = 8
        elif i < 2:
            patch_size = 4
        else:
            patch_size = 2

        if i >= 2 or nbits == 0:
            coarseerror = 1.0
            reduceerror = 1.0
            mask_out = torch.zeros(mask.size()).to(mask.device)
            nbits = 0
        else:

            # The regular variant - just generate a mask once and move on
            init, mask_query_count, mask_out, nbits, tr_score = \
                          generate_mask(model, img_small_torch, img_torch, lbl_v, mask, tar, tar_large, lbl_t, pt_file, scorefile,
                                        heatmap, coarseerror, reduceerror, num_xforms = num_xforms_mask,
                                        net_size = net_size, model_type = model_type, patch_size = patch_size, stride_factor = 2)#8)#noise_size // 8)
            nbits = int(nbits)

            mask_out_np = mask_out.permute(1, 2, 0).numpy()
            cv2.imwrite(output_base + 'masks/' + out_str_base + '_' + str(nbits) + '_bits_' + str(tr_score) + '_tr_' + str(i) + '_iter_' + tag + '.png', mask_out_np * 255)

            init_np = init.permute(1, 2, 0).numpy()
            init_np = cv2.cvtColor(init_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_base + 'inits/' + out_str_base + '_' + str(nbits) + '_bits_' + str(tr_score) + '_tr_' + str(i) + '_iter_' + tag + '.png', init_np * 255)

            print("Predicted label for initialized example: ", model.predict(convert2Network(init, net_size = net_size)))

            total_query_count += mask_query_count

        if model_type == 'OpenALPRBorder':
            img_torch = tmp
            img_small_torch = tmp_small
            mask_inner = mask_out.clone()

        ##################################### STAGE 2: Boost / optimize noise within mask ##################################
        if model_type == 'OpenALPRBorder':
            mask_out = mask_outer + mask_out
            mask_out = torch.where(mask_out > 0.5, torch.ones(mask_out.size()), torch.zeros(mask_out.size()))


        args = parsearguments.getarguments()
        num_boost_iters = 5
        adversarial, tr_score, perturb, boost_query_count, next_theta, adv_small = boost(model, img_torch, lbl_v, mask_out, tar, lbl_t, beta = beta, iterations = num_boost_iters, pt_file = pt_file, num_xforms = num_xforms_boost, net_size = net_size, model_type = model_type, eta = eta, bt = bt, budget_factor = 2000, init_theta = next_theta)

        total_query_count += boost_query_count

        if model_type == 'OpenALPRBorder':
            tar_large = adversarial.clone()
            tar = torch.from_numpy(cv2.resize(tar_large.clone().permute(1, 2, 0).numpy(), (noise_x, noise_y))).permute(2, 0, 1)

        adversarial_np = adversarial.permute(1, 2, 0).numpy()
        adversarial_np = cv2.cvtColor(adversarial_np, cv2.COLOR_RGB2BGR)
        
    
        cv2.imwrite(output_base + 'boosted/' + out_str_base + '_' + str(nbits) + '_bits_' + str(tr_score) + '_tr_' + str(i) + '_iter_' + tag + '.png', adversarial_np * 255)

    perturb_np = perturb.permute(1, 2, 0).numpy()
    perturb_np = cv2.cvtColor(perturb_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_base + 'perturbations/' + out_str_base + '_' + str(nbits) + 'bits_' + str(tr_score) + '_tr_' + str(i) + '_iter_' + tag + '.png', perturb_np * 255)

    print("--------------------------------")
    print("Attack Completed.")
    print("Final transform_robustness:", tr_score)
    print("Final number of pixels:", nbits)
    print("Final number of queries:", total_query_count)
    return


# Wrapper for attacking a GTSRB model
def attack_GTSRB(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta= 1, num_xforms_mask = 100, num_xforms_boost = 1000, net_size = 32, noise_size = 32):
    net = GTSRBNet()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        
    net.eval()

    model = net.module if torch.cuda.is_available() else net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('GTSRB/checkpoint_us.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    attack_network(model, img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta, num_xforms_mask, num_xforms_boost, net_size, noise_size, model_type = 'GTSRB')


# Wrapper for attacking an ALPR (automatic license plate recognizer) model, border attack style

def attack_openalpr_border(img_v, img_t, mask_outer, mask_inner, vic_lp, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta= 1, num_xforms_mask = 100, num_xforms_boost = 1000, net_size = 224, noise_size = 32):
    model = OpenALPRBorderNet(vic_lp)

    attack_network(model, img_v, img_t, mask, 0, 1, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta, num_xforms_mask, num_xforms_boost, net_size, noise_size, model_type = 'OpenALPRBorder', mask_outer=mask_outer, mask_inner=mask_inner)


# Wrapper for attacking an ALPR (automatic license plate recognizer) model
def attack_openalpr(img_v, img_t, mask, vic_lp, tar_lp, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta= 1, num_xforms_mask = 100, num_xforms_boost = 1000, net_size = 224, noise_size = 32):
    args = parsearguments.getarguments()

    t = 1
    model = OpenALPRNet(vic_lp, tar_lp)

    attack_network(model, img_v, img_t, mask, 0, t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta, num_xforms_mask, num_xforms_boost, net_size, noise_size, model_type = 'OpenALPR')


# Start of execution path
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    network = 'GTSRB'
    args = parsearguments.getarguments()
    network = args.network
    img_v = args.img_v
    img_t = args.img_t
    mask = args.mask
    lbl_v = args.lbl_v
    lbl_t = args.lbl_t
    pt_file = args.pt_file
    scorefile = args.scorefile
    heatmap = args.heatmap
    num_xforms_boost = args.boost_transforms
    num_xforms_mask = args.mask_transforms
    vic_lp = args.vic_license_plate
    tar_lp = args.tar_license_plate
    mask_outer = args.border_outer
    mask_inner = args.border_inner
    beta = 1.0 
    

    timestart = time.time()

    if network == 'GTSRB':
        attack_GTSRB(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, args.coarse_error, args.reduce_error, beta, num_xforms_mask, num_xforms_boost, net_size = 32, noise_size = 32)
    elif network == 'OpenALPRBorder':
        attack_openalpr_border(img_v, img_t, mask_outer, mask_inner, vic_lp, pt_file, scorefile, heatmap, args.coarse_error, args.reduce_error, beta, num_xforms_mask, num_xforms_boost, net_size = 500, noise_size = 250)
    else:
        attack_openalpr(img_v, img_t, mask, vic_lp, tar_lp, pt_file, scorefile, heatmap, args.coarse_error, args.reduce_error, beta, num_xforms_mask, num_xforms_boost, net_size = 459, noise_size = 459)

    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))
