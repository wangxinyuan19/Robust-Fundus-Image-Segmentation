from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch.nn.functional as F
from torchvision.transforms import functional as FT
import torch
import pickle
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        if train:
            image_patch_list = sorted(os.listdir(path_Data+'training_pro/img_patch/'))
            mask_patch_list = sorted(os.listdir(path_Data+'training_pro/gt_patch/'))
            self.data = []
            for i in range(len(image_patch_list)):
                img_patch_path = path_Data+'training_pro/img_patch/' + image_patch_list[i]
                mask_patch_path = path_Data+'training_pro/gt_patch/' + mask_patch_list[i]
                self.data.append([img_patch_path, mask_patch_path])
            # self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data+'test_pro/img/'))
            masks_list = sorted(os.listdir(path_Data+'test_pro/gt/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'test_pro/img/' + images_list[i]
                mask_path = path_Data+'test_pro/gt/' + masks_list[i]
                self.data.append([img_path, mask_path])
            # self.transformer = config.test_transformer
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        with open(file=img_path, mode='rb') as file:
            img = torch.from_numpy(pickle.load(file)).float()
        with open(file=msk_path, mode='rb') as file:
            msk = torch.from_numpy(pickle.load(file)).float()

        seed = torch.seed()
        torch.manual_seed(seed)
        img = self.transforms(img)
        torch.manual_seed(seed)
        msk = self.transforms(msk)
            
        return img, msk

    def __len__(self):
        return len(self.data)
    


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Fix_RandomRotation(object):

    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        return FT.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

