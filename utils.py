import math
import torch 
import time
import parsearguments

def run_predictions_untargeted(model, imgs, label):
    batch_size = 256

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_tensor = torch.zeros((batch_size, imgs[0].size()[0], imgs[0].size()[1], imgs[0].size()[2]))
    num_successes = 0
    lbl_tensor = (torch.ones((batch_size)) * label).long().to(device)

    count = 0
    for i, img in enumerate(imgs):
        img_tensor[count, :, :, :] = img
        count += 1
        if count == batch_size or i == len(imgs) - 1:
            if count < batch_size:
                img_tensor = img_tensor[:count, :, :, :]
                lbl_tensor = lbl_tensor[:count]
            preds = model.predict(img_tensor)
            num_successes += (preds != lbl_tensor).sum().item()
            count = 0

    return (1.0 - num_successes * 1.0 / len(imgs)), len(imgs)


def run_predictions_targeted(model, imgs, target, victim=None):
    batch_size = 256

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_tensor = torch.zeros((batch_size, imgs[0].size()[0], imgs[0].size()[1], imgs[0].size()[2]))
    num_successes = 0
    tar_tensor = (torch.ones((batch_size)) * target).long().to(device)
    if victim is not None:
        vic_tensor = (torch.ones((batch_size)) * victim).long().to(device)
        vic_successes = 0

    count = 0
    for i, img in enumerate(imgs):
        max_y = min(img.size()[1], img_tensor.size()[2])
        max_x = min(img.size()[2], img_tensor.size()[3])
        img_tensor[count, :, :max_y, :max_x] = img[:, :max_y, :max_x]
        count += 1
        if count == batch_size or i == len(imgs) - 1:
            if count < batch_size:
                img_tensor = img_tensor[:count, :, :, :]
                tar_tensor = tar_tensor[:count]
                vic_tensor = vic_tensor[:count]
            preds = model.predict(img_tensor)
            num_successes += (preds == tar_tensor).sum().item()
            if victim is not None:
                vic_successes += (preds == vic_tensor).sum().item()
            count = 0

    return (1.0 - num_successes * 1.0 / len(imgs)), len(imgs)


def run_predictions(model, imgs, label, target = None):
    """ imgs: set of transfomed images
        label:  is the victim's label
        target: is the target's label
    """
    if target is None:
        return run_predictions_untargeted(model, imgs, label)
    else:
        return run_predictions_targeted(model, imgs, target, label)
