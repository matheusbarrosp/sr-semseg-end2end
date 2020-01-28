import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange

from skimage import io
from skimage import transform
import sys

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(img_in, img_tar, img_bic, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale
    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))
    img_bic = img_bic.crop((ty,tx,ty + tp, tx + tp))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_bic, info_patch

def augment(img_in, img_tar, img_bic, img_msk, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_bic = ImageOps.flip(img_bic)
        img_msk = ImageOps.flip(img_msk)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_bic = ImageOps.mirror(img_bic)
            img_msk = ImageOps.mirror(img_msk)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_bic = img_bic.rotate(180)
            img_msk = img_msk.rotate(180)
            info_aug['trans'] = True
            
    return img_in, img_tar, img_bic, img_msk, info_aug
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, train_dir, patch_size, sr_patch_size, upscale_factor, num_classes, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.train_images = [(join(data_dir, train_dir, x), join(data_dir, train_dir+'_msk', x)) for x in 
            listdir(join(data_dir,train_dir)) if is_image_file(x)]
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.patch_size = patch_size
        self.sr_patch_size = sr_patch_size
        self.upscale_factor = upscale_factor
        self.num_classes = num_classes
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        img_path, msk_path = self.train_images[index]
        img = load_img(img_path)
        _, file_name = os.path.split(img_path)
        lr = img.resize((int(img.size[0]/self.upscale_factor),int(img.size[1]/self.upscale_factor)), Image.BICUBIC)       
        bicubic = rescale_img(lr, self.upscale_factor)
        msk = load_img(msk_path)
        """
        ix = -1
        iy = -1
        if lr.size[0] == self.patch_size and lr.size[1] == self.patch_size:
            ix = 0
            iy = 0

        lr, img, bicubic, _ = get_patch(lr,img,bicubic,self.patch_size, self.upscale_factor, ix, iy)
        """
        if self.data_augmentation:
            lr, img, bicubic, msk, _ = augment(lr, img, bicubic, msk)
        
        img = np.array(img)
        lr = np.array(lr)
        bicubic = np.array(bicubic)
        msk = np.array(msk)
        msk = msk[:,:,0]
        #msk = io.imread(msk_path)
        
        #img = img.astype(np.float32)
        msk = msk.astype(np.int64)
        
        if self.num_classes == 2:
            msk[msk < 127] = 0
            msk[msk >= 127] = 1
            
        if 'grss' in self.data_dir:
            msk = msk-1
        
        """
        # Statistical normalization
        for i in range(img.shape[2]):
            img[:,:,i] = (img[:,:,i] - img[:,:,i].mean())
        """
        
        if self.transform:
            lr = self.transform(lr)
            bicubic = self.transform(bicubic)
            img = self.transform(img)
        
        msk = torch.from_numpy(msk)
        
        return lr, bicubic, img, msk, file_name

    def __len__(self):
        return len(self.train_images)

"""
class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, upscale_factor, num_classes, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.test_images = [(join(data_dir, 'test', x), join(data_dir, 'test_msk', x)) for x in listdir(join(data_dir, 'test')) if 
                            is_image_file(x)]
        self.data_dir = data_dir
        self.upscale_factor = upscale_factor
        self.num_classes = num_classes
        self.transform = transform

    def __getitem__(self, index):
        img_path, msk_path = self.test_images[index]
        img = load_img(img_path)
        _, file = os.path.split(self.img_path)
        bicubic = rescale_img(img, self.upscale_factor)
        
        img = np.array(img)
        bicubic = np.array(bicubic)
        msk = io.imread(msk_path)        
        
        img = img.astype(np.float32)
        msk = msk.astype(np.int64)
        
        if self.num_classes == 2:
            msk[msk < 127] = 0
            msk[msk >= 127] = 1
            
        if 'grss' in self.data_dir:
            msk = msk-1
            
        # Statistical normalization
        #if self.in_channels == 1:
        #        img = (img - img.mean()) / img.std()
        #    else:
        #        for i in range(img.shape[2]):
        #            img[:,:,i] = (img[:,:,i] - img[:,:,i].mean())
        
        if self.transform:
            img = self.transform(img)
            bicubic = self.transform(bicubic)
            
        msk = torch.from_numpy(msk)
            
        return img, bicubic, msk, file
      
    def __len__(self):
        return len(self.test_images)
"""

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, test_dir, upscale_factor, num_classes, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.val_images = [(join(data_dir, test_dir, x), join(data_dir, test_dir+'_msk', x)) for x in 
            listdir(join(data_dir, test_dir)) if is_image_file(x)]
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.upscale_factor = upscale_factor
        self.num_classes = num_classes
        self.transform = transform


    def __getitem__(self, index):
        img_path, msk_path = self.val_images[index]
        img = load_img(img_path)
        _, file = os.path.split(img_path)
        lr = img.resize((int(img.size[0]/self.upscale_factor),int(img.size[1]/self.upscale_factor)), Image.BICUBIC)       
        bicubic = rescale_img(lr, self.upscale_factor)
        msk = load_img(msk_path)
        
        img = np.array(img)
        lr = np.array(lr)
        bicubic = np.array(bicubic)
        msk = np.array(msk)
        msk = msk[:,:,0]
        #msk = io.imread(msk_path)
        
        #img = img.astype(np.float32)
        msk = msk.astype(np.int64)        
        
        if self.num_classes == 2:
            msk[msk < 127] = 0
            msk[msk >= 127] = 1
            
        if 'grss' in self.data_dir:
            msk = msk-1
        
        """
        # Statistical normalization
        for i in range(img.shape[2]):
            img[:,:,i] = (img[:,:,i] - img[:,:,i].mean())
        """

        if self.transform:
            lr = self.transform(lr)
            bicubic = self.transform(bicubic)
            img = self.transform(img)
        
        msk = torch.from_numpy(msk)
        
        return lr, bicubic, img, msk, file


    def __len__(self):
        return len(self.val_images)


class DatasetFromFolderValidation(data.Dataset):
    def __init__(self, data_dir, upscale_factor, num_classes, transform=None):
        super(DatasetFromFolderValidation, self).__init__()
        self.val_images = [(join(data_dir, 'val', x), join(data_dir, 'val_msk', x)) for x in listdir(join(data_dir, 'val')) if is_image_file(x)]
        self.data_dir = data_dir
        self.upscale_factor = upscale_factor
        self.num_classes = num_classes
        self.transform = transform


    def __getitem__(self, index):
        img_path, msk_path = self.val_images[index]
        img = load_img(img_path)
        _, file = os.path.split(img_path)
        lr = img.resize((int(img.size[0]/self.upscale_factor),int(img.size[1]/self.upscale_factor)), Image.BICUBIC)       
        bicubic = rescale_img(lr, self.upscale_factor)
        msk = load_img(msk_path)
        
        img = np.array(img)
        lr = np.array(lr)
        bicubic = np.array(bicubic)
        msk = np.array(msk)
        msk = msk[:,:,0]
        #msk = io.imread(msk_path)
        
        #img = img.astype(np.float32)
        msk = msk.astype(np.int64)        
        
        if self.num_classes == 2:
            msk[msk < 127] = 0
            msk[msk >= 127] = 1
            
        if 'grss' in self.data_dir:
            msk = msk-1
        
        """
        # Statistical normalization
        for i in range(img.shape[2]):
            img[:,:,i] = (img[:,:,i] - img[:,:,i].mean())
        """

        if self.transform:
            lr = self.transform(lr)
            bicubic = self.transform(bicubic)
            img = self.transform(img)
        
        msk = torch.from_numpy(msk)
        
        return lr, bicubic, img, msk, file


    def __len__(self):
        return len(self.val_images)
