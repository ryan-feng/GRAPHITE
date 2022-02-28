import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import subprocess

class OpenALPRBorderNet():
    def __init__(self, vic_lp):
        self.vic_lp = vic_lp

    def run_alpr_detect(self, image):
        img_np = image.permute(1, 2, 0).numpy()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_path = 'temp.jpg'
        cv2.imwrite(img_path, img_np * 255.0)
        a = subprocess.run(["alpr", "-c", "us", img_path], stdout=subprocess.PIPE)
        b = subprocess.run(["awk", "{getline; print $2;}"], input=a.stdout, stdout=subprocess.PIPE)
        c = subprocess.run(["head", "-1"], input=b.stdout, stdout=subprocess.PIPE)
        return c.stdout.decode('utf-8')[:-1]
        

    def predict(self, image, target=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            image_batch = image.clone()
            if len(image.size()) < 4:
                image_batch = image_batch.unsqueeze(0)
            predict = torch.zeros((image_batch.size()[0]), dtype=torch.long)
            for i in range(predict.size()[0]):
                lp = self.run_alpr_detect(image_batch[i])
                if lp != self.vic_lp:
                    predict[i] = 1
                else:
                    predict[i] = 0

        predict = predict.to(device)

        if len(image.size()) < 4:
            return predict[0].item()
        return predict
