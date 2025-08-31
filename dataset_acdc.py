# This is not the original from the Git repo of Swin-Unet. It has been edited to incorporate additional augmentations.

import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
#from timm.data.auto_augment import rand_augment_transform # edited here
import PIL # edited here
from PIL import Image # edited here
from PIL import ImageEnhance # edited here
#from matplotlib import pyplot as plt # edited here


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

#edited here
def identity(image, label):
    return image, label

def shear_x(image, label):
    #input = image
    cx = -.05 + .1*random.random()
    tfm_mat = np.array([[1., cx, 0.],
                        [0, 1., 0],
                        [0, 0, 1.]])
    matrix = np.linalg.inv(tfm_mat)

    image = ndimage.affine_transform(image, matrix)

    label = ndimage.affine_transform(label, matrix)

    return image, label

def shear_y(image, label):
    #input = image
    cy = -.05 + .1*random.random()
    tfm_mat = np.array([[1, 0, 0],
                        [cy, 1, 0],
                        [0, 0, 1]])
    matrix = np.linalg.inv(tfm_mat)

    image = ndimage.affine_transform(image, matrix)

    label = ndimage.affine_transform(label, matrix)

    return image, label

def translate_x(image, label):
    #input = image
    vx = random.randint(-image.shape[0]/2, image.shape[0]/2)
    tfm_mat = np.array([[1, 0, vx],
                        [0, 1, 0],
                        [0, 0, 1]])
    matrix = np.linalg.inv(tfm_mat)

    image = ndimage.affine_transform(image, matrix)

    label = ndimage.affine_transform(label, matrix)

    return image, label

def translate_y(image, label):
    #input = image
    vy = random.randint(-image.shape[1]/2, image.shape[1]/2)
    tfm_mat = np.array([[1, 0, 0],
                        [0, 1, vy],
                        [0, 0, 1]])
    matrix = np.linalg.inv(tfm_mat)

    image = ndimage.affine_transform(image, matrix)

    label = ndimage.affine_transform(label, matrix)

    return image, label

def scale_xy(image, label):
    #input = image
    cx = 0.9 + 0.2*random.random() #
    cy = cx
    tfm_mat = np.array([[cx, 0, 0],
                        [0, cy, 0],
                        [0, 0, 1]])
    matrix = np.linalg.inv(tfm_mat)

    image = ndimage.affine_transform(image, matrix)

    label = ndimage.affine_transform(label, matrix)

    return image, label

def scale_y(image, label):
    #input = image
    cy = 0.5 + random.random() #
    tfm_mat = np.array([[1, 0, 0],
                        [0, cy, 0],
                        [0, 0, 1]])
    matrix = np.linalg.inv(tfm_mat)

    image = ndimage.affine_transform(image, matrix)

    label = ndimage.affine_transform(label, matrix)

    return image, label

def autocon(image, label):
    image2 = Image.fromarray(np.uint8(image*255)) # PIL image uint8 0-255
    low = 20*random.random()
    high = 20*random.random()
    image_autocon = PIL.ImageOps.autocontrast(image2, cutoff = (low, high))
    image_autocon_arr = np.array(image_autocon) #ndarray unit8 0-255
    image_autocon_arr2 = image_autocon_arr.astype('float32') #ndarray f32 0-255
    image_autocon_arr3 = image_autocon_arr2/255 #ndarray f32 0-1

    image = image_autocon_arr3

    return image, label

def equalize_(image, label):
    image2 = Image.fromarray(np.uint8(image*255)) # PIL image uint8 0-255
    image_eq = PIL.ImageOps.equalize(image2)
    image_eq_arr = np.array(image_eq) #ndarray unit8 0-255
    image_eq_arr2 = image_eq_arr.astype('float32') #ndarray f32 0-255
    image_eq_arr3 = image_eq_arr2/255 #ndarray f32 0-1

    image = image_eq_arr3

    return image, label

def sharpness_(image, label):
    image2 = Image.fromarray(np.uint8(image*255)) # PIL image uint8 0-255
    enhancer = ImageEnhance.Sharpness(image2)
    factor = 2*random.random()
    image_sharpened = enhancer.enhance(factor)
    image_sharpened_arr3 = np.array(image_sharpened).astype('float32')/255 #ndarray unit8 0-255 -> ndarray f32 0-255 -> ndarray f32 0-1

    image = image_sharpened_arr3

    return image, label
#

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # edited here
        r = random.random()
        #print('r =', r)
        n_ = 7
        if r >= 0 and r < 1/n_ :
            image, label = identity(image, label)
            #print('r =', r)
        elif r >= 1/n_ and r < 2/n_ :
            image, label = random_rot_flip(image, label)
            #print('r =', r)
        elif r >= 2/n_ and r < 3/n_ :
            image, label = random_rotate(image, label)
            #print('r =', r)
        elif r >= 3/n_ and r < 4/n_ :
            if random.random() > .5:
                image, label = translate_x(image, label)
                #print('r =', r)
            else:
                image, label = translate_y(image, label)
        elif r >= 4/n_ and r < 5/n_ :
            image, label = autocon(image, label)
            #print('r =', r)
        elif r >= 5/n_ and r < 6/n_ :
            image, label = equalize_(image, label)
            #print('r =', r)
        elif r >= 6/n_ and r < 1 :
            image, label = sharpness_(image, label)
        # edited here
        

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class ACDC_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
