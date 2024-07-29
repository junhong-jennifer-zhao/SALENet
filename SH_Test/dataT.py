import os
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import util
import os.path as osp
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

import pickle5 as pickle
import imageio
import matplotlib.pyplot as plt
# imageio.plugins.freeimage.download()
import shutil
import random
import re


class extract_mesh():
    def __init__(self, h=128, w=256, ln=64):
        self.h, self.w = h, w
        steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
        steradian = np.sin(steradian / h * np.pi)
        steradian = np.tile(steradian.transpose(), (w, 1))
        steradian = steradian.transpose()
        self.steradian = steradian[..., np.newaxis]

        y_ = np.linspace(0, np.pi, num=h)  # + np.pi / h
        x_ = np.linspace(0, 2 * np.pi, num=w)  # + np.pi * 2 / w
        X, Y = np.meshgrid(x_, y_)
        Y = Y.reshape((-1, 1))
        X = X.reshape((-1, 1))
        xyz = util.polar_to_cartesian((X, Y))
        xyz = xyz.reshape((h, w, 3))  # 128, 256, 3
        xyz = np.expand_dims(xyz, axis=2)
        self.xyz = np.repeat(xyz, ln, axis=2)
        self.anchors = util.sphere_points(ln)

        dis_mat = np.linalg.norm((self.xyz - self.anchors), axis=-1)
        self.idx = np.argsort(dis_mat, axis=-1)[:, :, 0]
        self.ln, _ = self.anchors.shape

    def compute(self, hdr):

        hdr = self.steradian * hdr
        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[..., 1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]
        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=-1)
        light = hdr * map
        remain = hdr * (1 - map)

        ambient = remain.sum(axis=(0, 1))    #mean(axis=0).mean(axis=0)
        anchors = np.zeros((self.ln, 3))

        for i in range(self.ln):
            mask = self.idx == i
            mask = np.expand_dims(mask, -1)
            anchors[i] = (light * mask).sum(axis=(0, 1))

        anchors_engergy = 0.3 * anchors[..., 0] + 0.59 * anchors[..., 1] + 0.11 * anchors[..., 2]
        distribution = anchors_engergy / anchors_engergy.sum()
        anchors_rgb = anchors.sum(0)   # energy
        intensity = np.linalg.norm(anchors_rgb)
        rgb_ratio = anchors_rgb / intensity
        # distribution = anchors / intensity

        parametric_lights = {"distribution": distribution,
                             'intensity': intensity,
                             'rgb_ratio': rgb_ratio,
                             'ambient': ambient}
        return parametric_lights, map

    def compute_depth(self,depth):
        depth = np.expand_dims(depth, axis=2)
        depth_intensity = self.steradian * depth

        anchors = np.zeros((self.ln, 1))

        for i in range(self.ln):
            mask = self.idx == i
            mask = np.expand_dims(mask, -1)
            anchors[i] = (depth_intensity * mask).sum(axis=(0, 1))

        distribution = anchors / anchors.max()

        return distribution

class ParameterTestDataset(Dataset):
    def __init__(self, test_dir):

        self.cropPath = test_dir
        # self.pairs = os.listdir(test_dir)
        # self.pairs = [img for img in os.listdir(test_dir) if img.endswith('.png') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img]
        self.pairs = [img for img in os.listdir(test_dir) if img.endswith('.png') and os.stat(osp.join(self.cropPath,img)).st_size>0 ]
        # self.pairs = [img for img in os.listdir(test_dir) if img.endswith('.jpg') or img.endswith('.jfif') and os.stat(osp.join(self.cropPath,img)).st_size>0 ]
        # self.pairs = [img for img in os.listdir(test_dir) if img.endswith('.jpg') and os.stat(osp.join(self.cropPath,img)).st_size>0 ]
    
        self.data_len = len(self.pairs)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        training_pair = {
            "crop": None,
            'name': None}

        pair = self.pairs[index]
    
        crop_path = osp.join(self.cropPath,pair)

        input = cv2.imread(crop_path)
        input = cv2.resize(input,(256,256))
        input = input[...,::-1]
        training_pair['crop'] = self.to_tensor(input.copy())
        training_pair['name'] = pair.replace('.png', '')
        # training_pair['name'] = pair.replace('.jpg', '').replace('.jfif','')

        return training_pair

    def __len__(self):
        return self.data_len


class ParameterTestDataset_exr(Dataset):
    def __init__(self, test_dir):

        self.cropPath = test_dir
        self.pairs = [img for img in os.listdir(test_dir) if img.endswith('.exr') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img]
    
        self.data_len = len(self.pairs)
        self.to_tensor = transforms.ToTensor()

        self.crop_tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        self.handle = util.PanoramaHandler()


    def __getitem__(self, index):
        training_pair = {
            "crop": None,
            'name': None}

        pair = self.pairs[index]
    
        crop_path = osp.join(self.cropPath,pair)
        input = self.handle.read_hdr(crop_path)
        input,alpha = self.crop_tone(input)

        if input is None:
            print('Wrong path:', crop_path)
            exit(-1)
        elif input.shape[0] != 256 or input.shape[1] != 256:
            input = cv2.resize(input, dsize=(256,256))
        training_pair['crop'] = self.to_tensor(input.copy())
        training_pair['name'] = pair.replace('.exr', '')

        return training_pair

    def __len__(self):
        return self.data_len
