import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import os
import sys
import traceback

DEBUG = True

dir_path = os.path.dirname(os.path.abspath(__file__))

def default_loader(img):
    return to_tensor(Image.open(img))

class custom_dset(Dataset):
    def __init__(self,
                 img_path,
                 txt_path,
                 img_transform=None,
                 loader=default_loader):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.img_list = [
                os.path.join(img_path, i.split()[0]) for i in lines
            ]
            self.label_list = [i.split()[1] for i in lines]
        self.img_transform = img_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        # img = self.loader(img_path)
        img = img_path
        if self.img_transform is not None:
            img = self.img_transform(img)
        img = img

        return img, label

    def __len__(self):
        return len(self.label_list)

img_path = [
'/home/pyapp/captcha_break/imgs/yzm_0.png',
'/home/pyapp/captcha_break/imgs/yzm_1.png',
'/home/pyapp/captcha_break/imgs/yzm_2.png',
'/home/pyapp/captcha_break/imgs/yzm_3.png',
'/home/pyapp/captcha_break/imgs/yzm_4.png',
'/home/pyapp/captcha_break/imgs/yzm_5.png',
'/home/pyapp/captcha_break/imgs/yzm_6.png',
'/home/pyapp/captcha_break/imgs/yzm_7.png',
'/home/pyapp/captcha_break/imgs/yzm_8.png',
'/home/pyapp/captcha_break/imgs/yzm_9.png',
'/home/pyapp/captcha_break/imgs/yzm_10.png',
'/home/pyapp/captcha_break/imgs/yzm_11.png',
'/home/pyapp/captcha_break/imgs/yzm_12.png',
'/home/pyapp/captcha_break/imgs/yzm_13.png',
'/home/pyapp/captcha_break/imgs/yzm_14.png',
'/home/pyapp/captcha_break/imgs/yzm_15.png',
'/home/pyapp/captcha_break/imgs/yzm_16.png',
'/home/pyapp/captcha_break/imgs/yzm_17.png',
'/home/pyapp/captcha_break/imgs/yzm_18.png',
'/home/pyapp/captcha_break/imgs/yzm_19.png',
'/home/pyapp/captcha_break/imgs/yzm_20.png',
'/home/pyapp/captcha_break/imgs/yzm_21.png',
'/home/pyapp/captcha_break/imgs/yzm_22.png',
'/home/pyapp/captcha_break/imgs/yzm_23.png',
'/home/pyapp/captcha_break/imgs/yzm_24.png',
'/home/pyapp/captcha_break/imgs/yzm_25.png',
'/home/pyapp/captcha_break/imgs/yzm_26.png',

]

def print_log(fuc,line,**kwargs):
    print('#'*3 + ' debug-start:', fuc, "-> print line:"+str(line), '#'*3 )
    for item in kwargs:
        print(" "*3, "==>"+str(item)+':', kwargs[item])
    print('#' * 3 + 'debug-end:', fuc, "-> print line:" + str(line), '#' * 3)
    print("\n")


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def collate_fn(batch):

    print_log(sys._getframe().f_code, sys._getframe().f_lineno, batch=batch) if DEBUG is True else None
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    print_log(sys._getframe().f_code, sys._getframe().f_lineno, batch=batch) if DEBUG is True else None

    img, label = zip(*batch)  # [(x1,x2),(x3,x4)] => (x1,x3)  (x2,x4)

    print_log(sys._getframe().f_code, sys._getframe().f_lineno,img=img,label=label) if DEBUG is True else None

    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = [0] * max_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)
        lens.append(len(label[i]))
    return img, pad_label, lens

img_path = '/home/pyapp/captcha_break/imgs/'
txt_path = dir_path + '/test_dataset_train.txt'
dataset = custom_dset(img_path,txt_path)
image, target = dataset[0]

image_pil = pil_loader(image)

print(image)

print_log(sys._getframe().f_code, sys._getframe().f_lineno,
          dataset=dataset,image=image,target=target,code=''.join([x for x in target]),
          image_pil=image_pil,image_tensor = to_tensor(image_pil)
          ) if DEBUG is True else None


exit(0)
print_log(sys._getframe().f_code, sys._getframe().f_lineno, dataset=dataset) if DEBUG is True else None
dataloader = DataLoader(dataset,batch_size=8, shuffle=False, collate_fn=collate_fn)

imgs,targets,lens = dataloader.__iter__().__next__()

print_log(sys._getframe().f_code, sys._getframe().f_lineno, imgs=imgs,targets=targets,lens=lens) if DEBUG is True else None
