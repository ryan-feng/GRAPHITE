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
    parser = argparse.ArgumentParser(description='Whitebox GRAPHITE')
    parser.add_argument('--out', default='outputs/whitebox/outputs', help='output dir')
    parser.add_argument('--gradient_shrink', default='N', help='Whether to shrink graident size or not. (Y/N)')
    parser.add_argument('--aligned', default='Y', help='Whether to align 244 and 32 patches to the decimal. (Y/N)')
    parser.add_argument('--victim', type=int, default=14, metavar='N', help='victim label.')
    parser.add_argument('--target', type=int, default=1, metavar='N', help='target label.')
    parser.add_argument('--step_size', type=float, default=0.0156862745, metavar='N', help='step size.')
    parser.add_argument('--num_xforms', type=int, default=100, metavar='N', help='num xforms.')
    parser.add_argument('--min_tr', type=float, default=0.8, metavar='N', help='min tr.')
    parser.add_argument('--iters', type=int, default=50, metavar='N', help='num iters.')
    parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed.')
    parser.add_argument('--patch_removal_size', type=float, default=-1, metavar='N', help='step size.')
    parser.add_argument('--patch_removal_interval', type=float, default=15.25, metavar='N', help='step size.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    # Attack parameters
    victim_label = args.victim
    target_label = args.target
    num_xforms = args.num_xforms
    step_size = args.step_size
    iterations = args.iters
    allowable_transform_robustness = args.min_tr
    pt_file = '../inputs/GTSRB/Points/' + str(victim_label) + '.csv'
    net_size = 32
    model_type = 'GTSRB'
    criterion = torch.nn.CrossEntropyLoss()
    pixels_per_round = 5000
    pixels_per_round_auto = True
    resize_interpolation = cv2.INTER_NEAREST
    patch_removal = True
    if args.patch_removal_size < 0:
        patch_removal_size = 30 if args.aligned != 'Y' else 30.5
    else:
        patch_removal_size = args.patch_removal_size
    patch_removal_interval = args.patch_removal_interval
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
    previous_mask = mask.clone()
    prev_transform_robustness = None
    prev_mask_size = None
    prev_attack = img.detach().clone()
    while True:
        adv_img = img.detach().clone()
        if rounds % 10 != 0 or rounds > 10 or True:
            adv_img = torch.where(mask > 0.5, prev_attack, adv_img)
        rand_start = torch.FloatTensor(3, 32, 32).uniform_(-8/255, 8/255)
        rand_start_np = rand_start.permute(1, 2, 0).numpy()
        rand_start_np = cv2.resize(rand_start_np, (244, 244))
        rand_start = torch.from_numpy(rand_start_np).permute(2, 0, 1).to(adv_img.device)
        adv_img = adv_img + mask * rand_start
        adv_img = torch.clamp(adv_img, 0, 1)
        start_img = adv_img.clone()
        adv_img.requires_grad = True
        final_avg_grad = None

        # Do EOT adversarial attack with current mask.
        loop_length = iterations if rounds > 0 else 500
        for iteration in range(loop_length):
            avg_grad = torch.zeros(adv_img.size()).cuda()
            for xform in xforms:
                xform_img = transform_wb(img.clone().unsqueeze(0), adv_img.unsqueeze(0), mask, xform, pt_file,
                                         net_size)
                if model_type == 'GTSRB':
                    xform_img = xform_img - 0.5
                logits = model(xform_img.cuda())
                loss = criterion(logits, (torch.ones(logits.shape[0]) * target_label).long().cuda())
                grad = torch.autograd.grad(loss, adv_img)[0]
                avg_grad += grad
            avg_grad /= len(xforms)
            num_queries += len(xforms)

            avg_grad_sign = avg_grad.clone()
            avg_grad_sign[torch.isnan(avg_grad_sign)] = 0
            grad_np = avg_grad_sign.permute(1, 2, 0).cpu().detach().numpy()
            if args.gradient_shrink == 'Y':
                grad_np = cv2.resize(grad_np, (32, 32))
            grad_np = np.sign(grad_np)
            if args.gradient_shrink == 'Y':
                grad_np = cv2.resize(grad_np, (244, 244))
            avg_grad_sign = torch.from_numpy(grad_np).permute(2, 0, 1).cuda()
            avg_grad_sign[torch.isnan(avg_grad_sign)] = 0
            adv_img = adv_img - mask * step_size * avg_grad_sign

            adv_img = adv_img.clamp(0, 1)
            transform_robustness = eval(img.detach().clone(), adv_img, mask, target_label, model, xforms, pt_file)
            num_queries += len(xforms)
            print("Iteration #{} | Transform_Robustness: {:.2f}%".format(iteration, (transform_robustness * 100)))
            final_avg_grad = avg_grad
            if transform_robustness >= allowable_transform_robustness:
                break

        # Reset mask in case attack failed (likely removed too many pixels at end of previous round).
        if not patch_removal and pixels_per_round_auto and transform_robustness < allowable_transform_robustness:
            mask = previous_mask.clone()
            adv_img = img.detach().clone()
            adv_img.requires_grad = True
            pixels_per_round //= 2
            continue

        # Remove low-impact pixel-patches or pixels in mask.
        if patch_removal:
            pert = adv_img - img
            final_avg_grad[torch.isnan(final_avg_grad)] = 0
            final_avg_grad = mask * final_avg_grad * pert
            pixelwise_avg_grads = torch.sum(torch.abs(final_avg_grad), dim=0)

            num_patches = 0
            if rounds < 10: num_patches = 4
            elif rounds < 20: num_patches = 4
            else: num_patches = 4
            for _ in range(num_patches):
                # Find minimum gradient patch and remove it.
                min_patch_grad = None
                min_patch_grad_idx = None
                for i in np.arange(0, pixelwise_avg_grads.shape[0] - patch_removal_size + 0.0001, patch_removal_interval):
                    for j in np.arange(0, pixelwise_avg_grads.shape[1] - patch_removal_size + 0.0001, patch_removal_interval):
                        patch_grad = pixelwise_avg_grads[int(round(i)):int(round(i + patch_removal_size)), int(round(j)):int(round(j + patch_removal_size))].sum()
                       
                        if mask[0, int(round(i)):int(round(i + patch_removal_size)), int(round(j)):int(round(j + patch_removal_size))].sum() > 0:
                            patch_grad = patch_grad / mask[0, int(round(i)):int(round(i + patch_removal_size)), int(round(j)):int(round(j + patch_removal_size))].sum()  #TODO1
                            if min_patch_grad is None or patch_grad.item() < min_patch_grad:
                                min_patch_grad = patch_grad.item()
                                min_patch_grad_idx = (i, j)
                if min_patch_grad_idx is None:
                    break
                i, j = min_patch_grad_idx
                mask[0, int(round(i)):int(round(i + patch_removal_size)), int(round(j)):int(round(j + patch_removal_size))] = 0
                mask[1, int(round(i)):int(round(i + patch_removal_size)), int(round(j)):int(round(j + patch_removal_size))] = 0
                mask[2, int(round(i)):int(round(i + patch_removal_size)), int(round(j)):int(round(j + patch_removal_size))] = 0
                print("Removed patch: {}".format((i, j)))
        else:
            for _ in range(pixels_per_round):
                # Create gradient saliency map.
                final_avg_grad = mask * final_avg_grad
                pixelwise_avg_grads = torch.sum(final_avg_grad, dim=0)
                pixelwise_avg_grads = torch.where(pixelwise_avg_grads == 0, torch.tensor(999999).float().cuda(),
                                                  pixelwise_avg_grads)
                pixelwise_avg_grads[torch.isnan(pixelwise_avg_grads)] = 999999

                # Find minimum gradient pixel and remove from allowed set.
                min_pixelwise_avg_grad_idx = (pixelwise_avg_grads == torch.min(pixelwise_avg_grads)).nonzero()[0]
                mask[(0,) + tuple(min_pixelwise_avg_grad_idx)] = 0
                mask[(1,) + tuple(min_pixelwise_avg_grad_idx)] = 0
                mask[(2,) + tuple(min_pixelwise_avg_grad_idx)] = 0
                print("Removed pixel: {}".format(tuple(min_pixelwise_avg_grad_idx)))

        this_mask = torch.where(mask > 0.5, 1, 0)
        if transform_robustness < allowable_transform_robustness:
            save_image(prev_attack, '{}/{}_{}.png'.format(results_dir, victim_label, target_label))
            save_image(previous_mask, '{}/{}_{}_mask.png'.format(results_dir, victim_label, target_label))
            print("Final transform_robustness:", prev_transform_robustness)
            print("Final mask size:", prev_mask_size)
            print("Final num queries:", num_queries)
            print("Final rounds:", rounds)
            break

        previous_mask = mask.clone()
        prev_attack = adv_img.detach().clone()
        prev_transform_robustness = transform_robustness
        prev_mask_size = int(round(this_mask.sum().item() / 3))

        rounds += 1
