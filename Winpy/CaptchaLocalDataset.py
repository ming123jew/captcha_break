import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from captcha.image import ImageCaptcha
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict



class CaptchaLocalDataset(Dataset):
    def __init__(self, label_img,characters, length, width, height, input_length, label_length):
        super(CaptchaLocalDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)

        self.label_img = label_img




    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_count = len( self.label_img )
        random_img = self.label_img[random.randint(0, img_count-1)]
        image = to_tensor(pil_loader(random_img[1]))
        target = torch.tensor([self.characters.find(x) for x in random_img[0]], dtype=torch.long)

        # print( random_img[0])
        # print([self.characters.find(x) for x in  random_img[0]])
        # print(target)
        # exit(0)

        input_length = torch.full(size=(1,), fill_value=self.input_length, dtype=torch.long)

        target_length = torch.full(size=(1,), fill_value=self.label_length, dtype=torch.long)
        return image, target, input_length, target_length