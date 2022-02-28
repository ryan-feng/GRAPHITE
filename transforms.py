from scipy.cluster.vq import vq
import math
import torch.nn as nn
import numpy as np
import cv2
import torch 
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
from perspective_transform import get_perspective_transform
from kornia.geometry.transform import resize
from kornia.enhance import adjust_gamma

def dist2pixels(dist, width, obj_width = 30):
    dist_inches = dist * 12
    return 1.0 * dist_inches * width / obj_width


def convert2Network(img, is_torch = True, net_size = 32, model_type = 'GTSRB'):
    if net_size == 32: # GTSRB
        if (is_torch):
            img = img.permute(1, 2, 0).detach().cpu().numpy()

        img = cv2.resize(img, (32, 32))
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = torch.clamp(img, 0.0, 1.0)
        assert(torch.max(img) <= 1.0 and torch.min(img) >= 0.0)
        if model_type == 'GTSRB':
            img = transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))(img)
            assert(torch.max(img) <= 0.5 and torch.min(img) >= -0.5)

    else: # ALPR
        if (is_torch):
            img = img.permute(1, 2, 0).numpy()

        if img.shape[0] > img.shape[1]: # tall and narrow
            net_x = net_size
            net_y = int(round(net_size / img.shape[1] * img.shape[0]))
        else: # wide and short or square 
            net_y = net_size 
            net_x = int(round(net_size / img.shape[0] * img.shape[1]))

        img = cv2.resize(img, (net_x, net_y))
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = torch.clamp(img, 0.0, 1.0)
        assert(torch.max(img) <= 1.0 and torch.min(img) >= 0.0)

    return img


def apply_transformation(img, mask, pert, angle, dist, gamma, blur, crop_percent, crop_off_x, crop_off_y, pt_file, net_size = 32, obj_width = 30, focal = 3, nps = False, model_type = 'GTSRB'):
    pert_np = pert.permute(1, 2, 0).detach().cpu().numpy()
    if nps:
        assert False

    if blur != 0:
        pert_np = cv2.GaussianBlur(pert_np, (blur, blur), 0)

    pert = torch.from_numpy(pert_np).permute(2, 0, 1)

    att = torch.where(mask > 0.5, pert, img)
    att_np = att.permute(1, 2, 0).detach().cpu().numpy()
    att_np = np.clip(att_np, 0.0, 1.0)

    dist = dist2pixels(dist, att_np.shape[1], obj_width)
    focal = dist2pixels(focal, att_np.shape[1], obj_width)
    att_np = get_perspective_transform(att_np, angle, att_np.shape[1], att_np.shape[0], focal, dist, crop_percent, crop_off_x, crop_off_y, pt_file)

    # Gamma
    att_uint = (att_np * 255.0).astype(np.uint8)
    table = np.empty((256), np.uint8)
    for i in range(256):
        table[i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    att_uint = cv2.LUT(att_uint, table)
    att_np = (att_uint / 255.0).astype(np.float32)
    att_np = np.clip(att_np, 0.0, 1.0)

    return convert2Network(att_np, False, net_size = net_size, model_type = model_type)


def get_transform_params(num_xforms, model_type = 'GTSRB', nps = False, baseline = False, patchguard = False):
    blur_kernels = [0, 3, 5, 7]
    if baseline:
        blur_kernels = [0, 3]
    transforms = []
    for _ in range(num_xforms):
        if model_type == 'GTSRB':
            angle = np.random.uniform(-50, 50)
            if baseline:
                angle = np.random.uniform(-40, 40)
            max_dist = 15.0
            if baseline:
                max_dist = 3.0
            dist = np.random.uniform(3.0, max_dist)
            gamma = np.random.uniform(1.0, 3.5)
            flip_flag = np.random.uniform(0.0, 1.0)
            if int(round(flip_flag)) == 1:
                gamma = 1.0 / gamma

            crop_percent = np.random.uniform(-0.03125, 0.03125)
            crop_off_x = np.random.uniform(-0.03125, 0.03125)
            crop_off_y = np.random.uniform(-0.03125, 0.03125)
            blur = blur_kernels[int(math.floor(np.random.uniform(0.0, 1.0) * len(blur_kernels)))]
            xform = (angle, dist, gamma, blur, crop_percent, crop_off_x, crop_off_y, 30, 3, nps)
            transforms.append(xform)
        elif model_type == 'CIFAR':
            blur_kernels = [0]
            angle = np.random.uniform(-30, 30)
            max_dist = 3.0
            if baseline:
                max_dist = 3.0
            dist = np.random.uniform(3.0, max_dist)
            gamma = np.random.uniform(1.0, 2.0)
            flip_flag = np.random.uniform(0.0, 1.0)
            if int(round(flip_flag)) == 1:
                gamma = 1.0 / gamma

            crop_percent = np.random.uniform(-0.03125, 0.03125)
            crop_off_x = np.random.uniform(-0.03125, 0.03125)
            crop_off_y = np.random.uniform(-0.03125, 0.03125)
            blur = blur_kernels[int(math.floor(np.random.uniform(0.0, 1.0) * len(blur_kernels)))]
            xform = (angle, dist, gamma, blur, crop_percent, crop_off_x, crop_off_y, 30, 3, nps)
            transforms.append(xform)
        else:
            angle = np.random.uniform(-15, 15)
            crop_percent = np.random.uniform(-0.03125, 0.03125)
            crop_off_x = np.random.uniform(-0.03125, 0.03125)
            crop_off_y = np.random.uniform(-0.03125, 0.03125)

            blur_kernels = [0, 3, 5]
            max_dist = 15.0
            gamma = np.random.uniform(1.0, 3.5)
            flip_flag = np.random.uniform(0.0, 1.0)
            if int(round(flip_flag)) == 1:
                gamma = 1.0 / gamma

            blur = blur_kernels[int(math.floor(np.random.uniform(0.0, 1.0) * len(blur_kernels)))]
            dist = np.random.uniform(10.0, max_dist)
            xform = (angle, dist, gamma, blur, crop_percent, crop_off_x, crop_off_y, 120, 10, nps)
            transforms.append(xform)
    return transforms


def add_noise(image, mask, lbd, theta, return_pert_and_mask = False, clip = True):
    mask_np = mask.permute(1, 2, 0).cpu().numpy()
    image_np = image.permute(1, 2, 0).cpu().numpy()
    theta_np = theta.permute(1, 2, 0).cpu().numpy()
   
    theta_large_np = cv2.resize(theta_np, (image.size()[2], image.size()[1]))
    comb = image_np + lbd * theta_large_np

    mask_large_np = cv2.resize(mask_np, (image.size()[2], image.size()[1]))
    mask_large_np = np.where(mask_large_np > 0.5, 1.0, 0.0)

    if clip == True:
        comb = np.clip(comb, 0, 1)
    if return_pert_and_mask:
        pert = np.where(mask_large_np > 0.5, comb, 0)
        return torch.from_numpy(comb).permute(2, 0, 1), torch.from_numpy(pert).permute(2, 0, 1), torch.from_numpy(mask_large_np).permute(2, 0, 1)
    return torch.from_numpy(comb).permute(2, 0, 1)


def get_transformed_images(image, mask, transforms, lbd, theta, pt_file = 'inputs/GTSRB/Points/14.csv', net_size = 32, model_type = 'GTSRB'):
    att, pert, mask = add_noise(image, mask, lbd, theta, True)

    if len(transforms) == 0:
        return [convert2Network(att, net_size = net_size, model_type = model_type)]

    images = []
    for transform in transforms:
        angle, dist, gamma, blur, crop_percent, crop_off_x, crop_off_y, obj_width, focal, nps = transform
        images.append(apply_transformation(image, mask, pert, angle, dist, gamma, blur, crop_percent, crop_off_x, crop_off_y, pt_file, net_size, obj_width, focal, nps, model_type = model_type))

    return images


def transform_wb(orig, att, mask, transform, pt_file = 'inputs/GTSRB/Points/14.csv', net_size = 32):
    angle, dist, gamma, blur, crop_percent, crop_off_x, crop_off_y, obj_width, focal, _ = transform
    att = torch.clamp(att, 0.0, 1.0)

    if blur != 0:
        kernel = np.zeros((blur * 2 - 1, blur * 2 - 1))
        kernel[blur - 1, blur - 1] = 1
        kernel = cv2.GaussianBlur(kernel, (blur, blur), 0)
        kernel = kernel[blur // 2:blur // 2 + blur, blur // 2:blur // 2 + blur]
        kernel = kernel[np.newaxis, :, :]
        kernel = np.repeat(kernel[np.newaxis, :, :, :], 3, axis=0)
        kernel_torch = torch.from_numpy(kernel)
        blur = nn.Conv2d(in_channels=3, out_channels=3,
                                    kernel_size=blur, groups=3, bias=False, padding=blur // 2)
        blur.weight.data = kernel_torch.to(att.dtype)
        blur.weight.requires_grad = False
        blur = blur.to(att.device)
        # the below is done this way to match the black box implementation
        pert = torch.where(mask > 0.5, att, torch.zeros(att.size()).to(mask.device))
        att = torch.where(mask > 0.5, blur(pert), orig)
    else:
        att = torch.where(mask > 0.5, att, orig)

    att = torch.clamp(att, 0.0, 1.0)
    dist = dist2pixels(dist, att.size()[2], obj_width)
    focal = dist2pixels(focal, att.size()[2], obj_width)
    att = get_perspective_transform(att, angle, att.size()[3], att.size()[2], focal, dist, crop_percent, crop_off_x, crop_off_y, pt_file, whitebox=True)

    # Gamma
    att = adjust_gamma(att, gamma)

    return convert2NetworkWB(att, True, net_size = net_size)

def convert2NetworkWB(img, is_torch = True, net_size = 32):
    orig_device = img.device
    if net_size == 32: # GTSRB
        img = resize(img, 32, align_corners=False)
        img = torch.clamp(img, 0.0, 1.0)
        assert(torch.max(img) <= 1.0 and torch.min(img) >= 0.0)

    else: # ALPR
        if img.size()[1] > img.size()[2]: # tall and narrow
            net_x = net_size
            net_y = int(round(net_size / img.size()[2] * img.size()[1]))
        else: # wide and short or square 
            net_y = net_size 
            net_x = int(round(net_size / img.size()[1] * img.size()[2]))

        img = resize(img, (net_y, net_x), align_corners=True)
        img = torch.clamp(img, 0.0, 1.0)
        assert(torch.max(img) <= 1.0 and torch.min(img) >= 0.0)

    return img.to(orig_device)
