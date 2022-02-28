import sys
sys.path.append("..")
import torch
import torchvision
from torch.autograd.functional import jacobian
from transforms import transform_wb, get_transform_params, convert2NetworkWB
from GTSRB.GTSRBNet import GTSRBNet
from tqdm import tqdm
import pickle
import cv2
import numpy as np
import random
from torchvision.utils import save_image
import os
import sys
import argparse

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def eval(orig_x, x, mask, target_label, model, xforms, pt_file, net_size=32):
    successes = 0
    preds = []
    for xform in xforms:
        with torch.no_grad():
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            transformed_x = transform_wb(orig_x, x, mask, xform, pt_file, net_size)
            transformed_x = transformed_x - 0.5
            logits = model(transformed_x)
            preds.append(torch.nn.functional.softmax(logits, dim=1)[0, target_label].item())
            success = int(logits.argmax(dim=1).detach().cpu().numpy()[0] == target_label)
            successes += success
    return successes / len(xforms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Whitebox patch')
    parser.add_argument('--out', default='outputs/whitebox_patch/outputs', help='output dir')
    parser.add_argument('--gradient_shrink', default='N', help='Whether to shrink gradient size or not. (Y/N)')
    parser.add_argument('--whole_attack_32', default='N', help='Whether to do the whole attack in 32x32 or not. (Y/N)')
    parser.add_argument('--limited_loc', default='Y', help='Whether to test just corners and center or all locations. (Y/N)')
    parser.add_argument('--victim', type=int, default=14, metavar='N', help='victim label.')
    parser.add_argument('--target', type=int, default=1, metavar='N', help='target label.')
    parser.add_argument('--step_size', type=float, default=0.0156862745, metavar='N', help='step size.')
    parser.add_argument('--num_xforms', type=int, default=100, metavar='N', help='num xforms.')
    parser.add_argument('--mask_size', type=int, metavar='N', help='mask size.')
    parser.add_argument('--iters', type=int, default=200, metavar='N', help='num iters.')
    args = parser.parse_args()

    # Attack parameters
    victim_label = args.victim
    target_label = args.target
    mask_length = args.mask_size
    num_xforms = args.num_xforms 
    step_size = args.step_size
    iterations = args.iters
    allowable_transform_robustness = 0.8
    pt_file = '../inputs/GTSRB/Points/' + str(victim_label) + '.csv'
    net_size = 32
    model_type = 'GTSRB'
    criterion = torch.nn.CrossEntropyLoss()
    pixels_per_round = 5000
    pixels_per_round_auto = True
    resize_interpolation = cv2.INTER_NEAREST
    patch_removal = True
    patch_removal_size = 30
    num_queries = 0

    results_dir = args.out
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Load victim img and starting mask.
    img = cv2.imread('../inputs/GTSRB/images/' + str(victim_label) + '.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32) / 255.0
    img_small = cv2.resize(img, (32, 32))
    img = torch.from_numpy(img).cuda().permute((2, 0, 1))
    img_small = torch.from_numpy(img_small).cuda().permute((2, 0, 1))
    img.requires_grad = True
    img_small.requires_grad = True

    tar_img = cv2.imread('../inputs/GTSRB/images/' + str(target_label) + '.png')
    tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
    tar_img = np.array(tar_img, dtype=np.float32) / 255.0
    tar_img_small = cv2.resize(tar_img, (32, 32))
    tar_img = torch.from_numpy(tar_img).cuda().permute((2, 0, 1))
    tar_img_small = torch.from_numpy(tar_img_small).cuda().permute((2, 0, 1))
    diff_small = tar_img_small - img_small
    diff_np = diff_small.permute(1, 2, 0).detach().cpu().numpy()
    diff_np = cv2.resize(diff_np, (244, 244))
    diff_large = torch.from_numpy(diff_np).cuda().permute((2, 0, 1))


    mask = cv2.imread('../inputs/GTSRB/Hulls/' + str(victim_label) + '.png')
    if args.whole_attack_32 != 'Y':
        mask = cv2.resize(mask, (244, 244))
    mask = np.where(mask > 128, 255, 0)
    mask = torch.from_numpy(mask).permute(2, 0, 1).cuda() / 255.0

    # Load transforms.
    xforms = get_transform_params(num_xforms, 'GTSRB')

    # Load net
    net = GTSRBNet()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
    net.eval()
    model = net.module if torch.cuda.is_available() else net
    checkpoint = torch.load('../GTSRB/checkpoint_us.tar')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Attack
    rounds = 0
    iteration = None
    transform_robustness = None
    best_mask = mask.clone()
    best_transform_robustness = -1
    best_attack = img.detach().clone() if args.whole_attack_32 != 'Y' else img_small.detach().clone()

    if args.whole_attack_32 != 'Y' and args.limited_loc == 'N':
        locations_x = [x for i in range(0, 244 - mask_length, 8)]
        locations_y = [y for i in range(0, 244 - mask_length, 8)]
    elif args.whole_attack_32 == 'Y' and args.limited_loc == 'N':
        locations_x = [x for i in range(0, 32 - mask_length, 1)]
        locations_y = [y for i in range(0, 32 - mask_length, 1)]
    elif args.whole_attack_32 != 'Y' and args.limited_loc == 'Y':
        locations_x = [0, 0, 244 - mask_length, 244 - mask_length, 122 - int(round(mask_length / 2))]
        locations_y = [0, 244 - mask_length, 0, 244 - mask_length, 122 - int(round(mask_length / 2))]
    elif args.whole_attack_32 == 'Y' and args.limited_loc == 'Y':
        locations_x = [0, 0, 32 - mask_length, 32 - mask_length, 16 - int(round(mask_length / 2))]
        locations_y = [0, 32 - mask_length, 0, 32 - mask_length, 16 - int(round(mask_length / 2))]
    else:
        assert False

    for i, j in zip(locations_y, locations_x):
        if args.whole_attack_32 == 'Y':
            mask = torch.zeros((3, 32, 32)).to(img.device)
        else:
            mask = torch.zeros((3, 244, 244)).to(img.device)
        mask[:, i:i+mask_length, j:j+mask_length] = 1
        adv_img = img.detach().clone()
        adv_img = torch.clamp(adv_img, 0, 1)
        start_img = adv_img.clone()
        adv_img.requires_grad = True

        # Do EOT adversarial attack with current mask.
        max_tr = -1
        best_img = None
        for iteration in range(iterations):
            avg_grad = torch.zeros(adv_img.size()).cuda()
            for xform in xforms:
                xform_img = transform_wb(img.detach().clone().unsqueeze(0), adv_img.unsqueeze(0), mask, xform, pt_file,
                                         net_size)
                if model_type == 'GTSRB':
                    xform_img = xform_img - 0.5
                logits = model(xform_img.cuda())
                loss = criterion(logits, (torch.ones(logits.shape[0]) * target_label).long().cuda())
                grad = torch.autograd.grad(loss, adv_img)[0]
                avg_grad += grad
            avg_grad /= len(xforms)

            avg_grad_sign = avg_grad.clone()
            avg_grad_sign[torch.isnan(avg_grad_sign)] = 0
            grad_np = avg_grad_sign.permute(1, 2, 0).cpu().detach().numpy()
            if args.gradient_shrink == 'Y' and args.whole_attack_32 != 'Y':
                grad_np = cv2.resize(grad_np, (32, 32))
            grad_np = np.sign(grad_np)
            if args.gradient_shrink == 'Y' and args.whole_attack_32 != 'Y':
                grad_np = cv2.resize(grad_np, (244, 244))
            avg_grad_sign = torch.from_numpy(grad_np).permute(2, 0, 1).cuda()
            avg_grad_sign[torch.isnan(avg_grad_sign)] = 0

            adv_img = adv_img - mask * step_size * avg_grad_sign
            adv_img = adv_img.clamp(0, 1)

            transform_robustness = eval(img.detach().clone().unsqueeze(0), adv_img, mask, target_label, model, xforms, pt_file)
            if transform_robustness > max_tr:
                max_tr = transform_robustness
                best_img = adv_img.clone()

        if max_tr > best_transform_robustness:
            save_image(best_img, '{}/{}_{}.png'.format(results_dir, victim_label, target_label))
            best_transform_robustness = max_tr
            best_mask = mask
            best_attack = adv_img.clone()

    print("Final transform_robustness:", best_transform_robustness)
