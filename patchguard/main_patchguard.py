import sys
sys.path.append("..")
import time
import random
import numpy as np
import cv2
import sys
from utils import run_predictions
import torch
import torchvision.transforms as transforms
from torchvision import models
from pg_boost import boost
from pg_generate_mask import generate_mask
from shutil import copyfile
import parsearguments
from scipy.cluster.vq import vq
from patchguard_predict import PatchGuard as PatchGuard
from patchguard_predict import PatchGuardNoDefense as PatchGuardNoDefense

def attack_network(model, img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror,
                   beta=1, num_xforms_mask=100, num_xforms_boost=1000, net_size=32, noise_size=32, model_type='CIFAR'):
    import getpass
    username = getpass.getuser()
    args = parsearguments.getarguments()
    coarse_mode = args.coarse_mode

    out_str_base = username + "_" + str(lbl_v) + '_' + str(lbl_t) + '_' + str(reduceerror) + '_' + str(
        coarseerror) + '_' + str(coarse_mode) + '_' + scorefile + "_";
    output_base = 'outputs_reproduce/'
    if model_type != 'CIFAR':
        output_base += model_type + '/'
    img = cv2.imread(img_v)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32) / 255.0
    img_small = cv2.resize(img, (noise_size, noise_size))
    img_torch = torch.from_numpy(img).permute(2, 0, 1)
    img_small_torch = torch.from_numpy(img_small).permute(2, 0, 1)

    mask_file = None
    if mask is not None:
        mask_file = mask
        mask_full = cv2.imread(mask)
        if mask_full.shape[0] != noise_size or mask_full.shape[1] != noise_size:
            mask = cv2.resize(mask_full, (noise_size, noise_size))

        else:
            mask = mask_full

        mask = np.where(mask > 128, 255, 0)
        mask = torch.from_numpy(mask).permute(2, 0, 1) / 255.0

    if img_t is not None:
        tar_large = cv2.imread(img_t)
        tar = cv2.resize(tar_large, (noise_size, noise_size))
        tar = cv2.cvtColor(tar, cv2.COLOR_RGB2BGR)
        tar = np.array(tar, dtype=np.float32) / 255.0

        tar = torch.from_numpy(tar).permute(2, 0, 1)
        tar_large = cv2.cvtColor(tar_large, cv2.COLOR_RGB2BGR)
        tar_large = np.array(tar_large, dtype=np.float32) / 255.0
        tar_large = torch.from_numpy(tar_large).permute(2, 0, 1).float()

    init, mask_query_count, mask_out, nbits, tr_score = \
        generate_mask(model, img_small_torch, img_torch, lbl_v, mask, tar, tar_large, lbl_t, pt_file, scorefile,
                      heatmap, coarseerror, reduceerror, num_xforms=num_xforms_mask,
                      net_size=net_size, model_type=model_type, patch_size=noise_size // 8)
    nbits = int(nbits)

    mask_out_np = mask_out.permute(1, 2, 0).numpy()
    cv2.imwrite(output_base + 'masks/' + out_str_base + '_' + str(nbits) + '_bits_' + str(tr_score) + '_tr.png',
                mask_out_np * 255)

    init_np = init.permute(1, 2, 0).numpy()
    init_np = cv2.cvtColor(init_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_base + 'inits/' + out_str_base + '_' + str(nbits) + '_bits_' + str(tr_score) + '_tr.png',
                init_np * 255)

    adversarial, tr_score, perturb, boost_query_count, adv_small = boost(model, img_torch, lbl_v, mask_out, tar,
                                                                           lbl_t, beta=beta, iterations=1000,
                                                                           pt_file=pt_file, num_xforms=num_xforms_boost,
                                                                           net_size=net_size, model_type=model_type)

    print(tr_score, nbits)
    adversarial_np = adversarial.permute(1, 2, 0).numpy()
    adversarial_np = cv2.cvtColor(adversarial_np, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_base + 'boosted/' + out_str_base + '_' + str(nbits) + '_bits_' + str(tr_score) + '_tr.png',
                adversarial_np * 255)

    perturb_np = perturb.permute(1, 2, 0).numpy()
    perturb_np = cv2.cvtColor(perturb_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        output_base + 'perturbations/' + out_str_base + '_' + str(nbits) + 'bits_' + str(tr_score) + 'tr.png',
        perturb_np * 255)

    print("--------------------------------")
    print("Attack Completed.")
    print("Final transform_robustness:", tr_score)
    print("Final number of pixels:", nbits)
    print("Final number of queries:", mask_query_count + boost_query_count)
    return


def attack_CIFAR(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror, beta=1,
                 num_xforms_mask=100, num_xforms_boost=1000, net_size=32, noise_size=32):
    net = PatchGuard()
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attack_network(net, img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, coarseerror, reduceerror,
                   beta, num_xforms_mask, num_xforms_boost, net_size, noise_size, model_type='CIFAR')


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    args = parsearguments.getarguments()
    network = 'CIFAR'
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
    beta = 1

    classes = list(range(10))
    indices = list(range(1, 11))

    idx = 0
    while True:
        lbl_v = random.sample(classes, 1)[0]
        idx_v = random.sample(indices, 1)[0]
        img_v = 'cifar/test/{}/{}.png'.format(lbl_v, idx_v)

        lbl_t = random.sample(classes, 1)[0]
        idx_t = random.sample(indices, 1)[0]
        if lbl_t == lbl_v:
            continue
        img_t = 'cifar/test/{}/{}.png'.format(lbl_t, idx_t)
        mask = '../plain_masks/mask.png'
        timestart = time.time()

        idx += 1
        print("=============================================================================")
        print("NEW EXPERIMENT: class {} idx {} to {} idx {}".format(lbl_v, idx_v, lbl_t, idx_t))
        print("=============================================================================")
        sys.stdout.flush()
        if network == 'CIFAR':
            attack_CIFAR(img_v, img_t, mask, lbl_v, lbl_t, pt_file, scorefile, heatmap, args.coarse_error,
                         args.reduce_error, beta, num_xforms_mask, num_xforms_boost, net_size=32, noise_size=32)

        timeend = time.time()
        print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))
        print("=============================================================================")
