import torch
import numpy as np
import csv
import cv2
import math
from kornia.geometry.transform import warp_perspective

def get_perspective_transform(img, angle, w, h, f, d, crop_percent, crop_off_x, crop_off_y, pt_file = 'inputs/GTSRB/Points/14.csv', whitebox=False):
    # img in numpy / cv2 form

    angle = math.radians(angle)
    x_cam_off = w / 2 - math.sin(angle) * d
    z_cam_off = -math.cos(angle) * d
    y_cam_off = h / 2

    R = np.array([[math.cos(angle), 0, -math.sin(angle), 0],
                  [0, 1, 0, 0],
                  [math.sin(angle), 0, math.cos(angle), 0],
                  [0, 0, 0, 1]])
    C = np.array([[1, 0, 0, -x_cam_off],
                  [0, 1, 0, -y_cam_off],
                  [0, 0, 1, -z_cam_off],
                  [0, 0, 0, 1]])

    RT = np.matmul(R, C)

    H = np.array([[f*RT[0, 0], f*RT[0, 1], f*RT[0, 3]],
                  [f*RT[1, 0], f*RT[1, 1], f*RT[1, 3]],
                  [RT[2, 0], RT[2, 1], RT[2, 3]]])


    x_off, y_off, crop_size = get_offset_and_crop_size(w, h, H, crop_percent, crop_off_x, crop_off_y, pt_file, f / d)

    M_aff = np.array([[1, 0, x_off],
                      [0, 1, y_off],
                      [0, 0, 1]])
    M = np.matmul(M_aff, H)

    if h > w: # tall and narrow
        crop_x = crop_size
        crop_y = int(round(crop_size / w * h))
    else: # wide and short or square 
        crop_y = crop_size 
        crop_x = int(round(crop_size / h * w))

    if not whitebox:
        dst = cv2.warpPerspective(img,M,(crop_x,crop_y), borderMode=cv2.BORDER_REPLICATE)
    else:
        dst = warp_perspective(img, torch.from_numpy(M).float().to(img.device).unsqueeze(0), (crop_y, crop_x), align_corners=True, padding_mode = 'border')

    return dst


def get_offset_and_crop_size(w, h, H, crop_percent, crop_off_x, crop_off_y, pt_file, ratio):
    pts = []
    if pt_file is not None and pt_file != '':
        with open(pt_file) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                pts.append(np.array([[float(row[0])], [float(row[1])], [float(row[2])]]))

        for i in range(len(pts)):
            pts[i][0,0] *= w / pts[i][2,0]
            pts[i][1,0] *= h / pts[i][2,0]
            pts[i][2,0] *= 1.0 / pts[i][2,0]

    else:
        pts = [ np.array([[0], [0], [1.0]]),
                np.array([[0], [h], [1.0]]),
                np.array([[w], [0], [1.0]]),
                np.array([[w], [h], [1.0]]) ]

    min_x = w
    min_y = h
    max_x = 0
    max_y = 0

    for pt in pts:
        new_pt = np.matmul(H, pt)
        new_pt /= new_pt[2, 0]

        if new_pt[0, 0] < min_x:
            min_x = new_pt[0, 0]
        if new_pt[0, 0] > max_x:
            max_x = new_pt[0, 0]
        if new_pt[1, 0] < min_y:
            min_y = new_pt[1, 0]
        if new_pt[1, 0] > max_y:
            max_y = new_pt[1, 0]

    if pt_file is not None and pt_file != '':
        if (max_x - min_x) / (max_y - min_y) < w / h: # result is tall and narrow
            diff_in_size = (max_y - min_y) / h * w - (max_x - min_x)
            orig_size = max_y - min_y if w > h else (max_y - min_y) / h * w
            crop_size = int(round(orig_size * (1.0 - crop_percent)))
            y_off = -min_y - int(round(crop_percent / 2 * orig_size))
            x_off = -min_x + int(round(diff_in_size / 2 - crop_percent / 2 * orig_size)) 

        else: # result is wide and short
            diff_in_size = (max_x - min_x) / w * h - (max_y - min_y)
            orig_size = max_x - min_x if h > w else (max_x - min_x) / w * h
            crop_size = int(round(orig_size * (1.0 - crop_percent)))
            x_off = -min_x - int(round(crop_percent / 2 * orig_size))
            y_off = -min_y + int(round(diff_in_size / 2 - crop_percent / 2 * orig_size)) 

        return x_off + crop_off_x * crop_size, y_off + crop_off_y * crop_size, crop_size 

    else:
        min_x -= (w * ratio - (max_x - min_x)) // 2
        min_y -= (h * ratio - (max_y - min_y)) // 2
 
        crop_size = int(round((1.0 - crop_percent) * min(w, h) * ratio))

        return -min_x - int(round(crop_percent / 2 * w * ratio)), -min_y - int(round(crop_percent / 2 * h * ratio)), crop_size
