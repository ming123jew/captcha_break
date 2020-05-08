import os

import os
import shutil
import numpy as np

def loadFiles():
    ret = []
    dir_path = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.abspath(dir_path+'/imgs/')
    print("==>访问:"+source_path)
    if os.path.exists(source_path):
        # root 所指的是当前正在遍历的这个文件夹的本身的地址
        # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if 'yzm_' in file:
                    src_file = os.path.join(root, file)
                    print(src_file)
                    ret.append(src_file)
    return ret

files = loadFiles()
file_train = files[:10]
file_test = files[10:]
np.save( "file_train.npy" ,file_train )
np.save( "file_test.npy" ,file_test )


#
# def pil_loader(path):
#     with open(path, 'rb') as f:
#         with Image.open(f) as img:
#             return img.convert('RGB')
#
# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)
#
# def default_loader(path):
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)
#
#
#
# class ImageFolder(data.Dataset):
#     """A generic data loader where the images are arranged in this way: ::
#
#         root/dog/xxx.png
#         root/dog/xxy.png
#         root/dog/xxz.png
#
#         root/cat/123.png
#         root/cat/nsdf3.png
#         root/cat/asd932_.png
#
#     Args:
#         root (string): Root directory path.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         loader (callable, optional): A function to load an image given its path.
#
#      Attributes:
#         classes (list): List of the class names.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         imgs (list): List of (image path, class_index) tuples
#     """
#
#
#     def __init__(self, root, transform=None, target_transform=None,
#                  loader=default_loader):
#         classes, class_to_idx = find_classes(root)
#         imgs = make_dataset(root, class_to_idx)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
#
#         self.root = root
#         self.imgs = imgs
#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, target) where target is class_index of the target class.
#         """
#         path, target = self.imgs[index]
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.imgs)