import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision import transforms

from captcha.image import ImageCaptcha
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict

import string
import os

from CaptchaLocalDataset import CaptchaLocalDataset
from Model import Model
from Utilss import train
from Utilss import valid
from Utilss import print_log,pil_loader
import sys

characters = string.digits + string.ascii_letters

def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s

def decode_target(sequence):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')

DEBUG = True

dir_path = os.path.dirname(os.path.abspath(__file__))
img_path_base = dir_path + "/../imgs/yzm_{}.png"
model = torch.load(dir_path + '/../ctc3_2.pth')
torch.no_grad()
# n = 26
# while n>0:
#     img_path = img_path_base.format(n)
#
#     img=pil_loader(img_path)
#     tensor_img = to_tensor(img)
#     output = model(tensor_img.unsqueeze(0).cpu())
#     output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
#
#     print_log(sys._getframe().f_code, sys._getframe().f_lineno,n=n,img_path=img_path,output_argmax=output_argmax,pred=decode(output_argmax[0]),output_argmax_0=output_argmax[0]) if DEBUG is True else None
#     n = n - 1


img_path = img_path_base.format(27)

img = pil_loader(img_path)
tensor_img = to_tensor(img)
output = model(tensor_img.unsqueeze(0).cpu())
output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)

print_log(sys._getframe().f_code, sys._getframe().f_lineno, img_path=img_path, output_argmax=output_argmax,
          pred=decode(output_argmax[0]), output_argmax_0=output_argmax[0]) if DEBUG is True else None