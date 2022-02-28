import numpy as np
import cv2
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from GTSRBNet import GTSRBNet
from GTSRBDataset import GTSRBDataset

def main(argv):
    img = argv[1]
    label = argv[2]
    target = argv[3]

    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA) # inter area is more stable for downsizing from extremely high res images
    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))(img)
    img_torch = torch.zeros((1, 3, 32, 32))
    img_torch[0, :, :, :] = img

    root = ''


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GTSRBNet()
    model.to(device)
    
    classes = []
    with open(root + 'class_semantics.txt') as f:
        for line in f:
            classes.append(line.strip())

    checkpoint = torch.load('checkpoint_us.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    correct = 0
    total = 0
    tar_top1 = 0
    tar_top2 = 0
    tar_top5 = 0
    lbl_top1 = 0
    lbl_top2 = 0
    lbl_top5 = 0
    with torch.no_grad():
        inputs = img_torch.to(device)
        labels = torch.tensor([int(label)]).to(device)
        labels = labels.long()
        outputs = model(inputs)
        conf, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        conf, lbls = torch.topk(F.softmax(outputs.data, 1), 5, 1)
        if int(label) in lbls[0]:
            lbl_top5 = 1
            if int(label) in lbls[0][:2]:
                lbl_top2 = 1
            if int(label) in lbls[0][0]:
                lbl_top1 = 1
        if int(target) in lbls[0]:
            tar_top5 = 1
            if int(target) in lbls[0][:2]:
                tar_top2 = 1
            if int(target) in lbls[0][0]:
                tar_top1 = 1
        print("Pred class: ", classes[predicted[0].item()], " Pred confidence: ", F.softmax(outputs.data, 1)[0][predicted[0].item()].item(), " Target confidence: ", F.softmax(outputs.data, 1)[0][int(target)].item(), " Stop confidence: ", F.softmax(outputs.data, 1)[0][int(label)].item())
        print("topk:", tar_top1, tar_top2, tar_top5, lbl_top1, lbl_top2, lbl_top5)
        print("conf:", F.softmax(outputs.data, 1)[0][int(target)].item(), F.softmax(outputs.data, 1)[0][int(label)].item())

    val_acc = 100.0 * correct / total
    print('Val accuracy: %.3f' % (val_acc))

if __name__ == '__main__':
    main(sys.argv)
