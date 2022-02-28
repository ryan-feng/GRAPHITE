import torch
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

if __name__ == '__main__':
    ds = datasets.CIFAR10(root='./cifar', train=False, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds)
    if not os.path.exists('cifar/test/'):
        os.makedirs('cifar/test')
        for i in range(10):
            os.makedirs('cifar/test/{}'.format(i))
    for x, y in tqdm(dl):
        y = y.cpu().numpy()[0]
        save_path = 'cifar/test/{}'.format(y)
        imgs_list = os.listdir(save_path)
        imgs_list.sort()
        if len(imgs_list) == 0:
            last_img_idx = 0
        else:
            last_img_idx = int(imgs_list[-1].split('.')[0])
        save_path = 'cifar/test/{}/{}.png'.format(y, last_img_idx + 1)
        save_image(x, save_path)
