import torch
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

if __name__ == '__main__':
    ds = datasets.CIFAR10(root='.', train=False, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds)
    counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if not os.path.exists('test/'):
        os.makedirs('test')
        for i in range(10):
            os.makedirs('test/{}'.format(i))
    for x, y in tqdm(dl):
        y = y.cpu().numpy()[0]
        save_path = 'test/{}'.format(y)
        imgs_list = os.listdir(save_path)
        imgs_list.sort()
        last_img_idx = counts[y]
        counts[y] += 1
        save_path = 'test/{}/{}.png'.format(y, last_img_idx)
        save_image(x, save_path)
