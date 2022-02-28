import sys
sys.path.append("..")
import torch
import torchvision
from torch.autograd.functional import jacobian
from transforms import transform_wb, get_transform_params, convert2NetworkWB, get_transformed_images, convert2Network
from utils import run_predictions
from GTSRB.GTSRBNet import GTSRBNet
from tqdm import tqdm
import pickle
import cv2
import numpy as np
import random
from torchvision.utils import save_image
import os
import argparse
from pprint import pprint
from prettytable import PrettyTable
import pickle
import seed


class Logger:
    def __init__(self, args):
        self.args = args

        self.log_dict = {
            'victim_label': args.victim_label,
            'target_label': args.target_label,
            'args': args,
            'round_results': {}
        }

    def update(self, current_round, transform_robustness, query_count, mask, adv_img):
        self.log_dict['round_results'][current_round] = {
            'query_count': query_count,
            'transform_robustness': transform_robustness,
            'mask': mask,
            'adv_img': adv_img
        }

    def save(self):
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, 'results_victim_label={}_target_label={}.pkl'.format(
                self.args.victim_label, self.args.target_label)), 'wb') as f:
            pickle.dump(self.log_dict, f)
        rounds = list(self.log_dict['round_results'].keys())
        rounds.sort()
        if len(rounds) > 0:
            last_round = rounds[-1]
            print("Final query count:", self.log_dict['round_results'][last_round]['query_count'])
            print("Final transform_robustness:", self.log_dict['round_results'][last_round]['transform_robustness'])
            print("Final mask_size:", torch.sum(self.log_dict['round_results'][last_round]['mask']).item() / 3)
            save_image(self.log_dict['round_results'][last_round]['adv_img'], os.path.join(self.args.log_dir, 'results_victim_label={}_target_label={}.png'.format(self.args.victim_label, self.args.target_label)))
            save_image(self.log_dict['round_results'][last_round]['mask'], os.path.join(self.args.log_dir, 'results_victim_label={}_target_label={}_mask.png'.format(self.args.victim_label, self.args.target_label)))


def compute_transform_robustness(img, delta, mask, model, xforms, xforms_pt_file, model_input_size, target_label):
    xform_imgs = get_transformed_images(img.detach().cpu(), mask, xforms, 1.0, delta.detach().cpu(),
                                        xforms_pt_file,
                                        net_size=model_input_size)
    neg_tr, qc = run_predictions(model, xform_imgs, -1, target_label)
    return 1 - neg_tr, qc


def main(args):
    print('-----------------------------------------------------------------------------------')
    print('Running baseline: {}'.format(__file__.split('.')[0]))
    print('-----------------------------------------------------------------------------------')
    assert args.model == 'GTSRB'
    args_table = PrettyTable(['Argument', 'Value'])
    for arg in vars(args):
        args_table.add_row([arg, getattr(args, arg)])
    print(args_table)
    logger = Logger(args)

    # Load victim img.
    img = cv2.imread(args.victim_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img, (32, 32))
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).cuda().permute((2, 0, 1)).cuda()
    img.requires_grad = True
    img_small = np.array(img_small, dtype=np.float32) / 255.0
    img_small = torch.from_numpy(img_small).cuda().permute((2, 0, 1)).cuda()
    img_small.requires_grad = True

    # Load target img.
    tar_img = cv2.imread(args.target_img_path)
    tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
    tar_img = cv2.resize(tar_img, (32, 32))
    tar_img = np.array(tar_img, dtype=np.float32) / 255.0
    tar_img = torch.from_numpy(tar_img).cuda().permute((2, 0, 1)).cuda()

    # Load starting mask.
    mask = cv2.imread(args.initial_mask_path)
    mask = np.where(mask > 128, 255, 0)
    mask = torch.from_numpy(mask).permute(2, 0, 1).cuda() / 255.0

    # Load transforms.
    xforms = get_transform_params(args.num_xforms, args.model, baseline = True)

    # Load net.
    net = GTSRBNet()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
    net.eval()
    model = net.module if torch.cuda.is_available() else net
    checkpoint = torch.load('../GTSRB/checkpoint_us.tar')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Attack.
    query_count = 0
    print('Initializing...')
    delta = mask * (tar_img - img_small)
    final_avg_grad = None
    for rounds in range(args.max_rounds):
        print('Beginning round: {}'.format(rounds))
        print('Stage 1: Boosting.')
        max_iteration_transform_robustness = 0
        max_iteration_delta = None
        for iteration in range(args.max_boosting_iterations):
            print('Boosting iteration: {}'.format(iteration))
            # Gradient estimation.
            gradient = torch.zeros_like(delta).cuda()
            eps, qc = compute_transform_robustness(img_small, delta, mask, model, xforms, args.xforms_pt_file,
                                            args.model_input_size, args.target_label)

            if max_iteration_delta is None:
                max_iteration_delta = delta
                max_iteration_transform_robustness = eps

            query_count += qc
            for _ in tqdm(range(10)):
                u = torch.randn(delta.size()).type(torch.FloatTensor).cuda() * mask
                u = u / torch.norm(u)
                ttt = delta + u
                eps_ttt, qc = compute_transform_robustness(img_small, ttt, mask, model, xforms, args.xforms_pt_file,
                                                    args.model_input_size, args.target_label)
                query_count += qc
                gradient += (eps_ttt - eps) * u
            final_avg_grad = 0.1 * gradient

            delta = delta - 500 * final_avg_grad

            transform_robustness, qc = compute_transform_robustness(img_small, delta, mask, model, xforms, args.xforms_pt_file,
                                                      args.model_input_size, args.target_label)
            query_count += qc

            if transform_robustness > max_iteration_transform_robustness:
                max_iteration_delta = delta
                max_iteration_transform_robustness = transform_robustness

            if torch.sum(final_avg_grad) == 0:
                break

        delta = max_iteration_delta
        transform_robustness = max_iteration_transform_robustness
        print('Found adversarial example | transform_robustness: {}%'.format(transform_robustness * 100))
        if transform_robustness < args.min_transform_robustness:
            print('Transform_Robustness of example is below allowable threshold, so stopping.')
            break

        delta_np = delta.permute(1, 2, 0).detach().cpu().numpy()
        delta_np_large = cv2.resize(delta_np, (img.size()[2], img.size()[1]))
        delta_large_torch = torch.from_numpy(delta_np_large).permute(2, 0, 1).to(img.device)
        adv_img_large = img + delta_large_torch
        logger.update(rounds, transform_robustness, query_count, mask, adv_img_large)

        if torch.sum(final_avg_grad) == 0:
            print('Gradient is zero, so cannot proceed to stage 2. End.')
            break


        print('Stage 2: Mask reduction.')
        pert = delta
        final_avg_grad[torch.isnan(final_avg_grad)] = 0
        final_avg_grad = mask * final_avg_grad * pert
        pixelwise_avg_grads = torch.sum(torch.abs(final_avg_grad), dim=0)

        # Find minimum gradient patches and remove them.
        for _ in range(args.patches_per_round):
            patch_removal_size = args.patch_removal_size
            patch_removal_interval = args.patch_removal_interval
            min_patch_grad = 99999999999999999
            min_patch_grad_idx = None
            for i in range(0, pixelwise_avg_grads.shape[0] - patch_removal_size, patch_removal_interval):
                for j in range(0, pixelwise_avg_grads.shape[1] - patch_removal_size, patch_removal_interval):
                    patch_grad = pixelwise_avg_grads[i:i + patch_removal_size, j:j + patch_removal_size].sum()
                    if mask[0, i:i + patch_removal_size, j:j + patch_removal_size].sum() > 0:
                        patch_grad = patch_grad / mask[0, i:i + patch_removal_size, j:j + patch_removal_size].sum()
                        if patch_grad.item() < min_patch_grad:
                            min_patch_grad = patch_grad.item()
                            min_patch_grad_idx = (i, j)
            i, j = min_patch_grad_idx
            mask[0, i:i + patch_removal_size, j:j + patch_removal_size] = 0
            mask[1, i:i + patch_removal_size, j:j + patch_removal_size] = 0
            mask[2, i:i + patch_removal_size, j:j + patch_removal_size] = 0
            print("Removed patch: {}".format((i, j)))
        delta = delta * mask
        print('-----------------------------------------------------------------------------------')
        if torch.sum(mask) == 0:
            break
    logger.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_label', type=int, default=1)
    parser.add_argument('--victim_label', type=int, default=14)
    parser.add_argument('--num_xforms', type=int, default=100)
    parser.add_argument('--step_size', type=float, default=2 / 255)
    parser.add_argument('--min_transform_robustness', type=float, default=0.80)
    parser.add_argument('--xforms_pt_file', type=str, default='../inputs/GTSRB/Points/14.csv')
    parser.add_argument('--model_input_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='GTSRB')
    parser.add_argument('--patch_removal_size', type=int, default=4)
    parser.add_argument('--patch_removal_interval', type=int, default=1)
    parser.add_argument('--patches_per_round', type=int, default=4)
    parser.add_argument('--max_rounds', type=int, default=999999999)
    parser.add_argument('--max_boosting_iterations', type=int, default=3)
    parser.add_argument('--log_dir', type=str, default='../logs/l0_and_boosting_no_tolerance')
    parser.add_argument('--victim_img_path', type=str, default='../inputs/GTSRB/images/14.png')
    parser.add_argument('--target_img_path', type=str, default='../inputs/GTSRB/images/1.png')
    parser.add_argument('--initial_mask_path', type=str, default='../plain_masks/mask.png')

    main(parser.parse_args())
