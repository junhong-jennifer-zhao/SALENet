"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import importlib
import torch.utils.data
import numpy as np
import pickle5 as pickle
# import pickle
import cv2
from PIL import Image
import util
import os.path as osp
import random
from SphericalHarmonics import sphericalHarmonics
from torchvision import transforms
import math

def horizontal_rotate_panorama(hdr_img, deg):
    shift = int(deg / 360.0 * hdr_img.shape[1])
    out_img = np.roll(hdr_img, shift=shift, axis=1)
    return out_img


class LavalIndoorDataset():

    def __init__(self, opt):
        self.opt = opt
        self.pairs = self.get_paths(opt)

        size = len(self.pairs)
        self.dataset_size = size

        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

        h, w = 128, 256
        steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
        steradian = np.sin(steradian / h * np.pi)
        steradian = np.tile(steradian.transpose(), (w, 1))
        steradian = steradian.transpose()
        self.steradian = steradian[..., np.newaxis]

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def get_paths(self, opt):
        # if opt.phase == 'train':
        dir = 'pkl/'
        pkl_dir = opt.dataroot + dir
        pairs = []
        nms = os.listdir(pkl_dir)

        for nm in nms:
            if nm.endswith('.pickle'):
                pkl_path = pkl_dir + nm
                warped_path = pkl_path.replace(dir, 'warped/')
                # warped_path = pkl_path.replace(dir, 'test/')
                warped_path = warped_path.replace('pickle', 'exr')
                # print (warped_path)
                if os.path.exists(warped_path):
                    pairs.append([pkl_path, warped_path])
        return pairs

    def __getitem__(self, index):

        ln = 128
        # read .exr image
        pkl_path, warped_path = self.pairs[index]

        handle = open(pkl_path, 'rb')
        pkl = pickle.load(handle)

        crop_path = warped_path.replace('warped', 'crop')
        crop = util.load_exr(crop_path)
        crop, alpha = self.tone(crop)
        crop = cv2.resize(crop, (128, 128))
        crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        hdr = util.load_exr(warped_path)

        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[...,
                                                       1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(
            np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]
        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=0)
        map = np.array(map).astype('uint8')
        map = torch.from_numpy(map).float()
        warped = np.transpose(hdr, (2, 0, 1))
        warped = torch.from_numpy(warped)
        warped = warped * alpha

        dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        intensity_gt = torch.from_numpy(np.array(
            pkl['intensity'])).float().cuda() * 0.01
        rgb_ratio_gt = torch.from_numpy(np.array(
            pkl['rgb_ratio'])).float().cuda()
        ambient_gt = torch.from_numpy(
            pkl['ambient']).float().cuda() / (128 * 256)
        depth_gt = torch.from_numpy(pkl['depth']).float().cuda()

        intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        dirs = util.sphere_points(ln)
        dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        size = torch.ones((1, ln)).cuda().float() * 0.0025
        light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        env_gt = util.convert_to_panorama(dirs, size, light_gt)

        depth_gt = depth_gt / depth_gt.max()
        depth_gt = depth_gt.view(1, ln, 1).repeat(1, 1, 3).view(1, ln * 3)
        env_depth = util.convert_to_panorama(dirs, size, depth_gt).squeeze()

        ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        env_gt = env_gt.view(3, 128, 256) + ambient_gt
        env_gt = env_gt * alpha
        env_gt = torch.cat([env_gt, env_depth])

        input_dict = {
            'input': env_gt,
            'crop': crop,
            'warped': warped,
            'map': map,
            'distribution': dist_gt,
            'intensity': intensity_gt,
            'name': pkl_path.split('/')[-1].split('.')[0]
        }

        return input_dict

    def __len__(self):
        return self.dataset_size


class MyTestLavalIndoorDataset():

    def __init__(self, opt):
        self.opt = opt

        self.pklPath = osp.join(r'G:\Dataset\IndoorHDRDataset\TestGen',
                                'test_data_pkl')
        self.cropPath = osp.join(r'G:\Dataset\IndoorHDRDataset\TestGen',
                                 'crop')
        self.warpedPath = osp.join(r'G:\Dataset\IndoorHDRDataset\TestGen',
                                   'warped')
        self.imgPath = r'G:\Dataset\IndoorHDRDataset\image'

        self.pairs = []
        handle = open('testlist.pickle', 'rb')
        gt_nms = pickle.load(handle)
        for nm in gt_nms:
            if os.path.exists(osp.join(self.imgPath, '{}.exr'.format(nm))):
                self.pairs.append(nm)
        # nms = os.listdir(r'G:\Dataset\IndoorHDRDataset\testCrop\warped')
        # self.pairs = []
        # for nm in nms:
        #     self.pairs.append(nm.replace('.exr',''))

        size = len(self.pairs)
        self.dataset_size = size

        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

        h, w = 128, 256
        steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
        steradian = np.sin(steradian / h * np.pi)
        steradian = np.tile(steradian.transpose(), (w, 1))
        steradian = steradian.transpose()
        self.steradian = steradian[..., np.newaxis]

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):

        ln = 128
        # read .exr image
        pair = self.pairs[index]
        # pkl_path, warped_path = self.pairs[index]

        rotate = random.randint(0, 6) * 60
        crop_path = osp.join(self.cropPath, '{}-{}.exr'.format(pair, rotate))
        gt_path = osp.join(self.pklPath, '{}-{}.pickle'.format(pair, rotate))
        warped_path = osp.join(self.warpedPath,
                               '{}-{}.exr'.format(pair, rotate))

        crop = util.load_exr(crop_path)

        handle = open(gt_path, 'rb')
        pkl = pickle.load(handle)

        crop, alpha = self.tone(crop)
        crop = cv2.resize(crop, (128, 128))
        crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        hdr = util.load_exr(warped_path)

        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[...,
                                                       1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(
            np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]
        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=0)
        map = np.array(map).astype('uint8')
        map = torch.from_numpy(map).float()

        warped = np.transpose(hdr, (2, 0, 1))
        warped = torch.from_numpy(warped)
        warped = warped * alpha

        dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        intensity_gt = torch.from_numpy(np.array(
            pkl['intensity'])).float().cuda()
        rgb_ratio_gt = torch.from_numpy(np.array(
            pkl['rgb_ratio'])).float().cuda()
        ambient_gt = torch.from_numpy(pkl['ambient']).float().cuda()

        intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        dirs = util.sphere_points(ln)
        dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        size = torch.ones((1, ln)).cuda().float() * 0.0025
        light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        env_gt = util.convert_to_panorama(dirs, size, light_gt)
        ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        env_gt = env_gt.view(3, 128, 256)  # + ambient_gt
        env_gt = env_gt * alpha

        input_dict = {
            'input': env_gt,
            'crop': crop,
            'warped': warped,
            'map': map,
            'distribution': dist_gt,
            'intensity': intensity_gt,
            'name': pair
        }

        return input_dict

    def __len__(self):
        return self.dataset_size


################### input SH_PART,SG_PART:Ground Truth ###################
##Train:SG+Depth+SH-->label_nc:9
class MyLavalIndoorDataset_addSH():

    def __init__(self, opt):
        self.opt = opt

        self.pklPath = osp.join(opt.dataroot, 'pkl')
        self.cropPath = osp.join(r'/media/common/zcy/datasets/train', 'crop')
        self.warpedPath = osp.join(r'/media/common/zcy/datasets/train',
                                   'warped')
        self.shPath = osp.join(r'/media/common/xjp', 'ambient_sh')

        self.pairs = []

        handle = open('trainlist.pickle', 'rb')
        gt_nms = pickle.load(handle)
        for nm in gt_nms:
            if os.path.exists(osp.join(self.warpedPath,
                                       '{}-0.exr'.format(nm))):
                self.pairs.append(nm)

        size = len(self.pairs)
        self.dataset_size = size

        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

        h, w = 128, 256
        steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
        steradian = np.sin(steradian / h * np.pi)
        steradian = np.tile(steradian.transpose(), (w, 1))
        steradian = steradian.transpose()
        self.steradian = steradian[..., np.newaxis]

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):

        ln = 128
        # read .exr image
        pair = self.pairs[index]
        # pkl_path, warped_path = self.pairs[index]

        rotate = random.randint(0, 180) * 2
        crop_path = osp.join(self.cropPath, '{}-{}.exr'.format(pair, rotate))
        gt_path = osp.join(self.pklPath, '{}-{}.pickle'.format(pair, rotate))
        warped_path = osp.join(self.warpedPath,
                               '{}-{}.exr'.format(pair, rotate))
        sh_path = osp.join(self.shPath, '{}-{}.npy'.format(pair, rotate))

        crop = util.load_exr(crop_path)

        handle = open(gt_path, 'rb')
        pkl = pickle.load(handle)

        crop, alpha = self.tone(crop)
        crop = cv2.resize(crop, (128, 128))
        crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        hdr = util.load_exr(warped_path)

        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[...,
                                                       1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(
            np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]
        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=0)
        map = np.array(map).astype('uint8')
        map = torch.from_numpy(map).float()

        warped = np.transpose(hdr, (2, 0, 1))
        warped = torch.from_numpy(warped)
        warped = warped * alpha

        dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        intensity_gt = torch.from_numpy(np.array(
            pkl['intensity'])).float().cuda() * 0.01
        rgb_ratio_gt = torch.from_numpy(np.array(
            pkl['rgb_ratio'])).float().cuda()
        ambient_gt = torch.from_numpy(
            pkl['ambient']).float().cuda() / (128 * 256)
        depth_gt = torch.from_numpy(pkl['depth']).float().cuda()

        intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        dirs = util.sphere_points(ln)
        dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        size = torch.ones((1, ln)).cuda().float() * 0.0025
        light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        env_gt = util.convert_to_panorama(dirs, size, light_gt)

        depth_gt = depth_gt / depth_gt.max()
        depth_gt = depth_gt.view(1, ln, 1).repeat(1, 1, 3).view(1, ln * 3)
        env_depth = util.convert_to_panorama(dirs, size, depth_gt).squeeze()

        # irradiance map
        sh = np.load(sh_path)  # (9,3)
        sh_map = sphericalHarmonics.shReconstructDiffuseMap(
            sh, width=256)  # (128,256,3)
        sh_map = np.transpose(sh_map, (2, 0, 1))  # (3, 128, 256)
        sh_map = torch.from_numpy(sh_map).float().cuda()

        ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        env_gt = env_gt.view(3, 128, 256) + ambient_gt
        env_gt = env_gt * alpha
        env_gt = torch.cat([env_gt, env_depth, sh_map])

        input_dict = {
            'input': env_gt,
            'crop': crop,
            'warped': warped,
            'map': map,
            'distribution': dist_gt,
            'intensity': intensity_gt,
            'name': pair + '_' + str(rotate)
        }

        return input_dict

    def __len__(self):
        return self.dataset_size


##Train:SG+SH-->label_nc:6
class MyLavalIndoorDataset_SHSG():

    def __init__(self, opt):
        self.opt = opt

        dataroot = r'/media/common/zcy/datasets/ExtIndoorDatabase'

        self.pklPath = osp.join(dataroot, 'pkl')
        self.warpedPath = osp.join(dataroot, 'warpedHDROutputs')
        self.cropPath = osp.join(dataroot, 'hdrInputs')
        self.shPath = osp.join(dataroot, 'ambient_sh')

        self.pairs = []

        handle = open(
            '/media/common/xjp/Illumination-Estimation-main/comparison/zcy_train_withinAbs_1.pickle',
            'rb')

        # handle = open('/media/common/xjp/Illumination-Estimation-main/Regression_SH_0_1/train_withinAbs_1.pickle', 'rb')
        gt_nms = pickle.load(handle)
        for nm in gt_nms:
            self.pairs.append(nm)

        size = len(self.pairs)
        self.dataset_size = size

        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

        h, w = 128, 256
        steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
        steradian = np.sin(steradian / h * np.pi)
        steradian = np.tile(steradian.transpose(), (w, 1))
        steradian = steradian.transpose()
        self.steradian = steradian[..., np.newaxis]

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):

        ln = 128
        pair = self.pairs[index]

        crop_path = osp.join(self.cropPath, pair)
        gt_path = osp.join(self.pklPath, pair.replace('.exr', '.pickle'))
        warped_path = osp.join(self.warpedPath, pair)
        sh_path = osp.join(self.shPath, pair.replace('.exr', '.npy'))

        crop = util.load_exr(crop_path)

        handle = open(gt_path, 'rb')
        pkl = pickle.load(handle)

        crop, alpha = self.tone(crop)
        crop = cv2.resize(crop, (128, 128))
        crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        hdr = util.load_exr(warped_path)

        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[...,
                                                       1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(
            np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]
        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=0)
        map = np.array(map).astype('uint8')
        map = torch.from_numpy(map).float()

        warped = np.transpose(hdr, (2, 0, 1))
        warped = torch.from_numpy(warped)
        warped = warped * alpha

        dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        intensity_gt = torch.from_numpy(np.array(
            pkl['intensity'])).float().cuda() * 0.01
        rgb_ratio_gt = torch.from_numpy(np.array(
            pkl['rgb_ratio'])).float().cuda()
        ambient_gt = torch.from_numpy(
            pkl['ambient']).float().cuda() / (128 * 256)

        intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        dirs = util.sphere_points(ln)
        dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        size = torch.ones((1, ln)).cuda().float() * 0.0025
        light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        env_gt = util.convert_to_panorama(dirs, size, light_gt)

        # irradiance map
        sh = np.load(sh_path)  # (9,3)
        sh_map = sphericalHarmonics.shReconstructDiffuseMap(
            sh, width=256)  # (128,256,3)
        sh_map = np.transpose(sh_map, (2, 0, 1))  # (3, 128, 256)
        sh_map = torch.from_numpy(sh_map).float().cuda()

        ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        env_gt = env_gt.view(3, 128, 256)  #+ ambient_gt
        env_gt = env_gt * alpha
        env_gt = torch.cat([env_gt, sh_map])

        input_dict = {
            'input': env_gt,
            'crop': crop,
            'warped': warped,
            'map': map,
            'distribution': dist_gt,
            'intensity': intensity_gt,
            'name': pair.replace('.exr', '')
        }

        return input_dict

    def __len__(self):
        return self.dataset_size

def resize_exr(img, res_x=512, res_y=512):

    theta, phi, move = 0.0, 0.0, 0.0
    img_x = img.shape[0]
    img_y = img.shape[1]

    theta = theta / 180 * math.pi
    phi = phi / 180 * math.pi

    axis_y = math.cos(theta)
    axis_z = math.sin(theta)
    axis_x = 0

    # theta rotation matrix
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    theta_rot_mat = np.array([[1, 0, 0], \
            [0, cos_theta, -sin_theta], \
            [0, sin_theta, cos_theta]], dtype=np.float32)

    # phi rotation matrix
    cos_phi = math.cos(phi)
    sin_phi = -math.sin(phi)
    phi_rot_mat = np.array([[cos_phi + axis_x**2 * (1 - cos_phi), \
            axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi, \
            axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi], \
            [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi, \
            cos_phi + axis_y**2 * (1 - cos_phi), \
            axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi], \
            [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi, \
            axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi, \
            cos_phi + axis_z**2 * (1 - cos_phi)]], dtype=np.float32)

    indx = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
    indy = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

    map_x = np.sin(indx * math.pi / res_x - math.pi / 2)
    map_y = np.sin(indy * (2 * math.pi)/ res_y) * np.cos(indx * math.pi / res_x - math.pi / 2)
    map_z = -np.cos(indy * (2 * math.pi)/ res_y) * np.cos(indx * math.pi / res_x - math.pi / 2)

    ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
            np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

    move_dir = np.array([0, 0, -1], dtype=np.float32)
    move_dir = theta_rot_mat.dot(move_dir)
    move_dir = phi_rot_mat.dot(move_dir)

    ind = theta_rot_mat.dot(ind)
    ind = phi_rot_mat.dot(ind)

    ind += np.tile(move * move_dir, (ind.shape[1], 1)).T

    vec_len = np.sqrt(np.sum(ind**2, axis=0))
    ind /= np.tile(vec_len, (3, 1))

    cur_phi = np.arcsin(ind[0, :])
    cur_theta = np.arctan2(ind[1, :], -ind[2, :])

    map_x = (cur_phi + math.pi/2) / math.pi * img_x
    map_y = cur_theta % (2 * math.pi) / (2 * math.pi) * img_y

    map_x = np.reshape(map_x, [res_x, res_y])
    map_y = np.reshape(map_y, [res_x, res_y])

    return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

##Train:SG+SH-->label_nc:6
class MyLavalIndoorDataset_SHSG_fromexr():

    def __init__(self, opt):
        self.opt = opt

        dataroot = '/local/scratch/zhaoj1/_locations_LavalTrain'

        self.cropPath = dataroot
        self.warpedPath = dataroot
        self.SG_path = '/local/scratch/zhaoj1/_locations_LavalTrain_SG'
        self.SH_path = dataroot

        self.pairs = []

        # handle = open(
        #     '/media/common/xjp/Illumination-Estimation-main/comparison/zcy_train_withinAbs_1.pickle',
        #     'rb')

        # # handle = open('/media/common/xjp/Illumination-Estimation-main/Regression_SH_0_1/train_withinAbs_1.pickle', 'rb')
        # gt_nms = pickle.load(handle) 
#9C4A0003_Panorama_hdr_Ref_view_ldr_1_80.png
#9C4A0003_Panorama_hdr_Ref_ibl_0_SG.exr
#9C4A0003_Panorama_hdr_Ref_ibl_0_SH.npy
#9C4A0003_Panorama_hdr_Ref_ibl_0.png.exr
        self.pairs =  [img for img in os.listdir(self.cropPath) if img.endswith('.exr') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img  \
            and os.path.exists(osp.join(self.SG_path, '{}_SG.exr'.format(img.replace('_80.exr', '').replace('_view_ldr','_ibl')))) \
            and os.path.exists(osp.join(self.SH_path, '{}_SH.npy'.format(img.replace('_80.exr', '').replace('_view_ldr','_ibl')))) \
            and os.path.exists(osp.join(self.warpedPath, '{}.png.exr'.format(img.replace('_80.exr', '').replace('_view_ldr','_ibl'))))]

        # for nm in gt_nms:
        #     self.pairs.append(nm)

        size = len(self.pairs)
        self.dataset_size = size

        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

        h, w = 128, 256
        steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
        steradian = np.sin(steradian / h * np.pi)
        steradian = np.tile(steradian.transpose(), (w, 1))
        steradian = steradian.transpose()
        self.steradian = steradian[..., np.newaxis]
        self.to_tensor = transforms.ToTensor()

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):

        ln = 128
        pair = self.pairs[index]

        # crop_path = osp.join(self.cropPath, pair)
        # gt_path = osp.join(self.pklPath, pair.replace('.exr', '.pickle'))
        # warped_path = osp.join(self.warpedPath, pair)
        # sh_path = osp.join(self.shPath, pair.replace('.exr', '.npy'))
        crop_path = osp.join(self.cropPath, pair)
        sg_path = osp.join(self.SG_path, '{}_SG.exr'.format(pair.replace('_80.exr', '').replace('_view_ldr','_ibl')))
        warped_path = osp.join(self.warpedPath, '{}.png.exr'.format(pair.replace('_80.exr', '').replace('_view_ldr','_ibl')))
        sh_path = osp.join(self.SH_path, '{}_SH.npy'.format(pair.replace('_80.exr', '').replace('_view_ldr','_ibl')))

        print("crop_path:",crop_path)
        print("sg_path:",sg_path)
        print("warped_path:",warped_path)
        print("sh_path:",sh_path)

        # crop_path = osp.join(self.cropPath, pair)

        crop = util.load_exr(crop_path)

        # handle = open(gt_path, 'rb')
        # pkl = pickle.load(handle)

        crop, alpha = self.tone(crop)
        crop = cv2.resize(crop, (128, 128))
        crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        hdr = util.load_exr(warped_path)
        hdr = resize_exr(abs(hdr), 128, 256) #gt hdr

        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[...,
                                                       1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(
            np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]
        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=0)
        map = np.array(map).astype('uint8')
        map = torch.from_numpy(map).float()

        warped = np.transpose(hdr, (2, 0, 1))
        warped = torch.from_numpy(warped)
        warped = warped * alpha

        #sg from pickle
        # dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        # intensity_gt = torch.from_numpy(np.array(
        #     pkl['intensity'])).float().cuda() * 0.01
        # rgb_ratio_gt = torch.from_numpy(np.array(
        #     pkl['rgb_ratio'])).float().cuda()
        # ambient_gt = torch.from_numpy(
        #     pkl['ambient']).float().cuda() / (128 * 256)

        # intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        # dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        # rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        # dirs = util.sphere_points(ln)
        # dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        # size = torch.ones((1, ln)).cuda().float() * 0.0025
        # light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        # env_gt = util.convert_to_panorama(dirs, size, light_gt)

        #sg from exr
        sg_map = cv2.imread(sg_path)
        # crop = cv2.resize(sg_map, (128, 128))
        sg_map = sg_map[..., ::-1]/50000
        sg_map = self.to_tensor(sg_map.copy()).float().cuda()
        # sg_map = sg_map* alpha
        # max_value = np.max(sg_map)
        # min_value = np.min(sg_map)
        # print("Max Value:", max_value)
        # print("Min Value:", min_value)
        # print("sg_map shape:",sg_map.shape)

        # irradiance map
        sh = np.load(sh_path)  # (9,3)
        sh_map = sphericalHarmonics.shReconstructDiffuseMap(
            sh, width=256)  # (128,256,3)
        sh_map = np.transpose(sh_map, (2, 0, 1))  # (3, 128, 256)
        sh_map = torch.from_numpy(sh_map).float().cuda()

        # ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        # env_gt = env_gt.view(3, 128, 256)  #+ ambient_gt
        # env_gt = env_gt * alpha
        env_gt = torch.cat([sg_map, sh_map])

        input_dict = {
            'input': env_gt,
            'crop': crop,
            'warped': warped,
            'map': map,
            # 'distribution': dist_gt,
            # 'intensity': intensity_gt,
            'name': pair.replace('.png', '')
        }

        return input_dict

    def __len__(self):
        return self.dataset_size


##Train:SG+SH-->label_nc:6
class MyLavalIndoorDataset_SHSG_fromexr_asg():

    def __init__(self, opt):
        self.opt = opt

        dataroot = '/local/scratch/zhaoj1/_locations_LavalTrain'

        self.cropPath = dataroot
        self.warpedPath = dataroot
        self.SG_path = '/local/scratch/zhaoj1/_locations_LavalTrain_ASG_noambient'
        self.SH_path = dataroot

        self.pairs = []

        # handle = open(
        #     '/media/common/xjp/Illumination-Estimation-main/comparison/zcy_train_withinAbs_1.pickle',
        #     'rb')

        # # handle = open('/media/common/xjp/Illumination-Estimation-main/Regression_SH_0_1/train_withinAbs_1.pickle', 'rb')
        # gt_nms = pickle.load(handle) 
#9C4A0003_Panorama_hdr_Ref_view_ldr_1_80.png
#9C4A0003_Panorama_hdr_Ref_ibl_0_SG.exr
#9C4A0003_Panorama_hdr_Ref_ibl_0_SH.npy
#9C4A0003_Panorama_hdr_Ref_ibl_0.png.exr
#9C4A0003_Panorama_hdr_Ref_ibl_0_ASG.exr
        self.pairs =  [img for img in os.listdir(self.cropPath) if img.endswith('.exr') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img  \
            and os.path.exists(osp.join(self.SG_path, '{}_ASG.exr'.format(img.replace('_80.exr', '').replace('_view_ldr','_ibl')))) \
            and os.path.exists(osp.join(self.SH_path, '{}_SH.npy'.format(img.replace('_80.exr', '').replace('_view_ldr','_ibl')))) \
            and os.path.exists(osp.join(self.warpedPath, '{}.png.exr'.format(img.replace('_80.exr', '').replace('_view_ldr','_ibl'))))]

        # for nm in gt_nms:
        #     self.pairs.append(nm)

        size = len(self.pairs)
        self.dataset_size = size

        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

        h, w = 128, 256
        steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
        steradian = np.sin(steradian / h * np.pi)
        steradian = np.tile(steradian.transpose(), (w, 1))
        steradian = steradian.transpose()
        self.steradian = steradian[..., np.newaxis]
        self.to_tensor = transforms.ToTensor()

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):

        ln = 128
        pair = self.pairs[index]

        # crop_path = osp.join(self.cropPath, pair)
        # gt_path = osp.join(self.pklPath, pair.replace('.exr', '.pickle'))
        # warped_path = osp.join(self.warpedPath, pair)
        # sh_path = osp.join(self.shPath, pair.replace('.exr', '.npy'))
        crop_path = osp.join(self.cropPath, pair)
        sg_path = osp.join(self.SG_path, '{}_ASG.exr'.format(pair.replace('_80.exr', '').replace('_view_ldr','_ibl')))
        warped_path = osp.join(self.warpedPath, '{}.png.exr'.format(pair.replace('_80.exr', '').replace('_view_ldr','_ibl')))
        sh_path = osp.join(self.SH_path, '{}_SH.npy'.format(pair.replace('_80.exr', '').replace('_view_ldr','_ibl')))

        print("crop_path:",crop_path)
        print("sg_path:",sg_path)
        print("warped_path:",warped_path)
        print("sh_path:",sh_path)

        # crop_path = osp.join(self.cropPath, pair)

        crop = util.load_exr(crop_path)

        # handle = open(gt_path, 'rb')
        # pkl = pickle.load(handle)

        crop, alpha = self.tone(crop)
        crop = cv2.resize(crop, (128, 128))
        crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        hdr = util.load_exr(warped_path)
        hdr = resize_exr(abs(hdr), 128, 256) #gt hdr

        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[...,
                                                       1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(
            np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]
        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=0)
        map = np.array(map).astype('uint8')
        map = torch.from_numpy(map).float()

        warped = np.transpose(hdr, (2, 0, 1))
        warped = torch.from_numpy(warped)
        warped = warped * alpha

        #sg from pickle
        # dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        # intensity_gt = torch.from_numpy(np.array(
        #     pkl['intensity'])).float().cuda() * 0.01
        # rgb_ratio_gt = torch.from_numpy(np.array(
        #     pkl['rgb_ratio'])).float().cuda()
        # ambient_gt = torch.from_numpy(
        #     pkl['ambient']).float().cuda() / (128 * 256)

        # intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        # dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        # rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        # dirs = util.sphere_points(ln)
        # dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        # size = torch.ones((1, ln)).cuda().float() * 0.0025
        # light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        # env_gt = util.convert_to_panorama(dirs, size, light_gt)

        #sg from exr
        sg_map = cv2.imread(sg_path)
        sg_map = cv2.resize(sg_map, (256, 128))
        sg_map = sg_map[..., ::-1]/50000
        sg_map = self.to_tensor(sg_map.copy()).float().cuda()
        # sg_map = sg_map* alpha
        # max_value = np.max(sg_map)
        # min_value = np.min(sg_map)
        # print("Max Value:", max_value)
        # print("Min Value:", min_value)
        # print("sg_map shape:",sg_map.shape)

        # irradiance map
        sh = np.load(sh_path)  # (9,3)
        sh_map = sphericalHarmonics.shReconstructDiffuseMap(
            sh, width=256)  # (128,256,3)
        sh_map = np.transpose(sh_map, (2, 0, 1))  # (3, 128, 256)
        sh_map = torch.from_numpy(sh_map).float().cuda()

        # ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        # env_gt = env_gt.view(3, 128, 256)  #+ ambient_gt
        # env_gt = env_gt * alpha
        env_gt = torch.cat([sg_map, sh_map])

        input_dict = {
            'input': env_gt,
            'crop': crop,
            'warped': warped,
            'map': map,
            # 'distribution': dist_gt,
            # 'intensity': intensity_gt,
            'name': pair.replace('.png', '')
        }

        return input_dict

    def __len__(self):
        return self.dataset_size

################### input SH_PART,SG_PART:Predictions ###################


##Test:SG+Depth+SH-->input:9
class TEST_SG_SH_Depth():

    def __init__(self, label_nc, SG_path, SH_path):

        self.cropPath = osp.join(r'/media/common/zcy/datasets/test', 'crop')
        self.warpedPath = osp.join(r'/media/common/zcy/datasets/test',
                                   'warped')
        self.SG_path = SG_path
        self.SH_path = SH_path
        self.label_nc = label_nc

        self.pairs = []

        handle = open('testlist.pickle', 'rb')
        gt_nms = pickle.load(handle)
        for nm in gt_nms:
            if os.path.exists(osp.join(self.warpedPath,
                                       '{}-0.exr'.format(nm))):
                self.pairs.append(nm)

        size = len(self.pairs)
        self.dataset_size = size

        sg_num = len(os.listdir(self.SG_path))
        sh_num = len(os.listdir(self.SH_path))

        assert sg_num == sh_num and sh_num == size, 'Size Mismatch!!!'

        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):

        ln = 128
        # read .exr image
        pair = self.pairs[index]
        # pkl_path, warped_path = self.pairs[index]

        # rotate = random.randint(0, 180) * 2
        rotate = 0

        crop_path = osp.join(self.cropPath, '{}-{}.exr'.format(pair, rotate))
        warped_path = osp.join(self.warpedPath,
                               '{}-{}.exr'.format(pair, rotate))

        gt_path = osp.join(self.SG_path, pair + '_0',
                           'pred-{}-{}.pickle'.format(pair, rotate))
        sh_path = osp.join(self.SH_path, pair + '_0', 'sh_pred.npy')

        crop = util.load_exr(crop_path)
        crop, alpha = self.tone(crop)
        crop = cv2.resize(crop, (128, 128))
        crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        hdr = util.load_exr(warped_path)
        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[...,
                                                       1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(
            np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]
        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=0)
        map = np.array(map).astype('uint8')
        map = torch.from_numpy(map).float()

        warped = np.transpose(hdr, (2, 0, 1))
        warped = torch.from_numpy(warped)
        warped = warped * alpha

        handle = open(gt_path, 'rb')
        pkl = pickle.load(handle)

        dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        intensity_gt = torch.from_numpy(np.array(
            pkl['intensity'])).float().cuda() * 0.01
        rgb_ratio_gt = torch.from_numpy(np.array(
            pkl['rgb_ratio'])).float().cuda()
        ambient_gt = torch.from_numpy(
            pkl['ambient']).float().cuda() / (128 * 256)
        depth_gt = torch.from_numpy(pkl['depth']).float().cuda()

        intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        dirs = util.sphere_points(ln)
        dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        size = torch.ones((1, ln)).cuda().float() * 0.0025
        light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        env_gt = util.convert_to_panorama(dirs, size, light_gt)

        depth_gt = depth_gt / depth_gt.max()
        depth_gt = depth_gt.view(1, ln, 1).repeat(1, 1, 3).view(1, ln * 3)
        env_depth = util.convert_to_panorama(dirs, size, depth_gt).squeeze()

        # irradiance map
        sh = np.load(sh_path)  # (9,3)
        sh_map = sphericalHarmonics.shReconstructDiffuseMap(
            sh, width=256)  # (128,256,3)
        sh_map = np.transpose(sh_map, (2, 0, 1))  # (3, 128, 256)
        sh_map = torch.from_numpy(sh_map).float().cuda()

        ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        env_gt = env_gt.view(3, 128, 256) + ambient_gt
        env_gt = env_gt * alpha

        assert self.label_nc == 3 or self.label_nc == 6 or self.label_nc == 9, 'wrong label_nc!!'

        if self.label_nc == 3:
            env_gt = env_gt
        elif self.label_nc == 6:
            env_gt = torch.cat([env_gt, env_depth])
        elif self.label_nc == 9:
            env_gt = torch.cat([env_gt, env_depth, sh_map])

        input_dict = {
            'input': env_gt,
            'crop': crop,
            'warped': warped,
            'map': map,
            'distribution': dist_gt,
            'intensity': intensity_gt,
            'name': pair + '_' + str(rotate)
        }

        return input_dict

    def __len__(self):
        return self.dataset_size


##Test:SG+SH -->input:6
class TEST_SG_SH():

    def __init__(self, label_nc, SG_path, SH_path, CROP_path):

        self.SG_path = SG_path
        self.SH_path = SH_path
        self.label_nc = label_nc

        self.cropPath = CROP_path
        # self.pairs = os.listdir(CROP_path)
        self.pairs =  [img for img in os.listdir(CROP_path) if img.endswith('.png') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img]
        # self.pairs =  [img for img in os.listdir(CROP_path) if img.endswith('.png') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_crop' in img]
        # self.pairs =  [img for img in os.listdir(CROP_path) if img.endswith('.jpg') and os.stat(osp.join(self.cropPath,img)).st_size>0 ]

        size = len(self.pairs)
        self.dataset_size = size
        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        self.to_tensor = transforms.ToTensor()

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):

        ln = 128
        pair = self.pairs[index]

        crop_path = osp.join(self.cropPath, pair)

        gt_path = osp.join(self.SG_path, pair.replace('.png', ''),
                           'sg_pred.pickle')
        sh_path = osp.join(self.SH_path, pair.replace('.png', ''),
                           'sh_pred.npy')

        # crop = util.load_exr(crop_path)
        # crop, alpha = self.tone(crop)
        # crop = cv2.resize(crop, (128, 128))
        # crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        crop = cv2.imread(crop_path)
        crop = cv2.resize(crop, (128, 128))
        crop = crop[..., ::-1]
        crop = self.to_tensor(crop.copy()).float().cuda()

        handle = open(gt_path, 'rb')
        pkl = pickle.load(handle)

        dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        intensity_gt = torch.from_numpy(np.array(
            pkl['intensity'])).float().cuda() * 0.01
        rgb_ratio_gt = torch.from_numpy(np.array(
            pkl['rgb_ratio'])).float().cuda()
        # ambient_gt = torch.from_numpy(
        #     pkl['ambient']).float().cuda() / (128 * 256)

        intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        dirs = util.sphere_points(ln)
        dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        size = torch.ones((1, ln)).cuda().float() * 0.0025
        light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        env_gt = util.convert_to_panorama(dirs, size, light_gt)

        # irradiance map
        sh = np.load(sh_path)  # (9,3)
        sh_map = sphericalHarmonics.shReconstructDiffuseMap(
            sh, width=256)  # (128,256,3)
        sh_map = np.transpose(sh_map, (2, 0, 1))  # (3, 128, 256)
        sh_map = torch.from_numpy(sh_map).float().cuda()

        # ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        env_gt = env_gt.view(3, 128, 256)  #+ ambient_gt

        # env_gt = env_gt * alpha
        # env_gt, alpha = self.tone(env_gt)

        assert self.label_nc == 6, 'wrong label_nc!!'

        env_gt = torch.cat([env_gt, sh_map])

        input_dict = {
            'input': env_gt,
            'crop': crop,
            'distribution': dist_gt,
            'intensity': intensity_gt,
            'name': pair.replace('.png', ''),
            'warped': torch.zeros(3, 128, 256),
            'map': torch.zeros(1, 128, 256)
        }

        return input_dict

    def __len__(self):
        return self.dataset_size

class TEST_SG_SH_fromexr():

    def __init__(self, label_nc, SG_path, SH_path, CROP_path):

        self.SG_path = SG_path
        self.SH_path = SH_path
        self.label_nc = label_nc

        self.cropPath = CROP_path
        # self.pairs = os.listdir(CROP_path)
        self.pairs =  [img for img in os.listdir(CROP_path) if img.endswith('.png') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img  \
            and os.path.exists(osp.join(self.SG_path, img.replace('.png', ''), 'pred-{}.exr'.format(img.replace('.png', '').replace('_','-')))) \
            and os.path.exists(osp.join(self.SH_path, img.replace('.png', ''), 'sh_pred.npy')) and 'AG8A9003' not in img and '9C4A5004' not in img and '9C4A0006' not in img \
             and '9C4A4987' not in img]

        size = len(self.pairs)
        self.dataset_size = size
        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        self.to_tensor = transforms.ToTensor()

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):
#9C4A0003_Panorama_hdr_Ref_view_ldr_1_80
        ln = 128
        pair = self.pairs[index]

        crop_path = osp.join(self.cropPath, pair)

        sg_path = osp.join(self.SG_path, pair.replace('.png', ''),
                           'pred-{}.exr'.format(pair.replace('.png', '').replace('_','-')))
        sh_path = osp.join(self.SH_path, pair.replace('.png', ''),
                           'sh_pred.npy')
        print("sg_path:",sg_path)

        # crop = util.load_exr(crop_path)
        # crop, alpha = self.tone(crop)
        # crop = cv2.resize(crop, (128, 128))
        # crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        crop = cv2.imread(crop_path)
        crop = cv2.resize(crop, (128, 128))
        crop = crop[..., ::-1]
        crop = self.to_tensor(crop.copy()).float().cuda()
        print("crop shape:",crop.shape)

        # handle = open(gt_path, 'rb')
        # pkl = pickle.load(handle)

        # dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        # intensity_gt = torch.from_numpy(np.array(
        #     pkl['intensity'])).float().cuda() * 0.01
        # rgb_ratio_gt = torch.from_numpy(np.array(
        #     pkl['rgb_ratio'])).float().cuda()
        # # ambient_gt = torch.from_numpy(
        # #     pkl['ambient']).float().cuda() / (128 * 256)

        # intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        # dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        # rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        # dirs = util.sphere_points(ln)
        # dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        # size = torch.ones((1, ln)).cuda().float() * 0.0025
        # light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        # env_gt = util.convert_to_panorama(dirs, size, light_gt)
        sg_map = cv2.imread(sg_path)
        sg_map = cv2.resize(sg_map, (256, 128))
        sg_map = sg_map[..., ::-1]/50000
        max_value = np.max(sg_map)
        min_value = np.min(sg_map)

        print("Max Value:", max_value)
        print("Min Value:", min_value)
        sg_map = self.to_tensor(sg_map.copy()).float().cuda()
        print("sg_map shape:",sg_map.shape)


        # irradiance map
        sh = np.load(sh_path)  # (9,3)
        print("sh coeefficient:",sh)
        max_value = np.max(sh)
        min_value = np.min(sh)
        print("Max Value SH Coeff:", max_value)
        print("Min Value SH Coeff:", min_value)
        sh_map = sphericalHarmonics.shReconstructDiffuseMap(
            sh, width=256)  # (128,256,3)
        sh_map = np.transpose(sh_map, (2, 0, 1))  # (3, 128, 256)

        max_value = np.max(sh_map)
        min_value = np.min(sh_map)

        print("Max Value SH:", max_value)
        print("Min Value SH:", min_value)
        print("sh_map shape:",sh_map.shape)

        sh_map = torch.from_numpy(sh_map).float().cuda()
        print("sh_map:",sh_map.shape) #[3, 128, 256]





        # ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        # env_gt = env_gt.view(3, 128, 256)  #+ ambient_gt

        # env_gt = env_gt * alpha
        # env_gt, alpha = self.tone(env_gt)

        assert self.label_nc == 6, 'wrong label_nc!!'

        env_gt = torch.cat([sg_map, sh_map])

        input_dict = {
            'input': env_gt,
            'crop': crop,
            'name': pair.replace('.png', ''),
            'warped': torch.zeros(3, 128, 256),
            'map': torch.zeros(1, 128, 256),
            'crop_path':crop_path
        }

        return input_dict

    def __len__(self):
        return self.dataset_size



class TEST_SG_SH_fromexr_SOTAcomparision():

    def __init__(self, label_nc, SG_path, SH_path, CROP_path):

        self.SG_path = SG_path
        self.SH_path = SH_path
        self.label_nc = label_nc

        self.cropPath = CROP_path
        self.pairs = [file for file in os.listdir(CROP_path) if "AG8A9403_Panorama_hdr_Ref_view_ldr_1_130_expo_1.0" not in file and "AG8A9270_Panorama_hdr_Ref_view_ldr_6_110_expo_1.0" not in file and 'AG8A9414_Panorama_hdr_Ref_view_ldr_5_90' not in file and 'AG8A9465_Panorama_hdr_Ref_view_ldr_7_130' not in file and 'Thumbs.db' not in file \
            and "AG8A9310_Panorama_hdr_Ref_view_ldr_3_70" not in file and 'AG8A9688_Panorama_hdr_Ref_view_ldr_3_70' not in file and '_crop' in file]
        # self.pairs = [file for file in os.listdir(CROP_path) if  file.endswith('.jpg') or file.endswith('.jfif') ]
        # self.pairs =  [img for img in os.listdir(CROP_path) if img.endswith('.png') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img  \
        #     and os.path.exists(osp.join(self.SG_path, img.replace('.png', ''), 'pred-{}.exr'.format(img.replace('.png', '').replace('_','-')))) \
        #     and os.path.exists(osp.join(self.SH_path, img.replace('.png', ''), 'sh_pred.npy')) and 'AG8A9003' not in img and '9C4A5004' not in img and '9C4A0006' not in img \
        #      and '9C4A4987' not in img]

        size = len(self.pairs)
        self.dataset_size = size
        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        self.to_tensor = transforms.ToTensor()

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):
#9C4A0003_Panorama_hdr_Ref_view_ldr_1_80  AG8A9228_Panorama_hdr_Ref_view_ldr_0_70_expo_1.0.png
        ln = 128
        pair = self.pairs[index]

        crop_path = osp.join(self.cropPath, pair)
        sg_path = osp.join(self.SG_path, pair,
                           'pred-{}.exr'.format(pair.replace('_','-')))
        sh_path = osp.join(self.SH_path, pair.replace('.png', ''),
                           'sh_pred.npy')
        # sg_path = osp.join(self.SG_path, pair.replace('.png', ''),
        #                    'pred-{}.exr'.format(pair.replace('.png', '').replace('_','-')))
        # sh_path = osp.join(self.SH_path, pair.replace('.png', ''),
        #                    'sh_pred.npy')
        # sg_path = osp.join(self.SG_path, pair.replace('.jpg', '').replace('.jfif',''),
        #                    'pred-{}.exr'.format(pair.replace('.jpg', '').replace('.jfif','').replace('_','-')))
        # sh_path = osp.join(self.SH_path, pair.replace('.jpg', '').replace('.jfif',''),
        #                    'sh_pred.npy')
        print("sg_path:",sg_path)

        # crop = util.load_exr(crop_path)
        # crop, alpha = self.tone(crop)
        # crop = cv2.resize(crop, (128, 128))
        # crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        crop = cv2.imread(crop_path)
        crop = cv2.resize(crop, (128, 128))
        crop = crop[..., ::-1]
        crop = self.to_tensor(crop.copy()).float().cuda()
        print("crop shape:",crop.shape)

        # handle = open(gt_path, 'rb')
        # pkl = pickle.load(handle)

        # dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        # intensity_gt = torch.from_numpy(np.array(
        #     pkl['intensity'])).float().cuda() * 0.01
        # rgb_ratio_gt = torch.from_numpy(np.array(
        #     pkl['rgb_ratio'])).float().cuda()
        # # ambient_gt = torch.from_numpy(
        # #     pkl['ambient']).float().cuda() / (128 * 256)

        # intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        # dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        # rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        # dirs = util.sphere_points(ln)
        # dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        # size = torch.ones((1, ln)).cuda().float() * 0.0025
        # light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        # env_gt = util.convert_to_panorama(dirs, size, light_gt)
        sg_map = cv2.imread(sg_path)
        sg_map = cv2.resize(sg_map, (256, 128))
        sg_map = sg_map[..., ::-1]/50000
        max_value = np.max(sg_map)
        min_value = np.min(sg_map)

        print("Max Value:", max_value)
        print("Min Value:", min_value)
        sg_map = self.to_tensor(sg_map.copy()).float().cuda()
        print("sg_map shape:",sg_map.shape)


        # irradiance map
        sh = np.load(sh_path)  # (9,3)
        print("sh coeefficient:",sh)
        max_value = np.max(sh)
        min_value = np.min(sh)
        print("Max Value SH Coeff:", max_value)
        print("Min Value SH Coeff:", min_value)
        sh_map = sphericalHarmonics.shReconstructDiffuseMap(
            sh, width=256)  # (128,256,3)
        sh_map = np.transpose(sh_map, (2, 0, 1))  # (3, 128, 256)

        max_value = np.max(sh_map)
        min_value = np.min(sh_map)

        print("Max Value SH:", max_value)
        print("Min Value SH:", min_value)
        print("sh_map shape:",sh_map.shape)

        sh_map = torch.from_numpy(sh_map).float().cuda()
        print("sh_map:",sh_map.shape) #[3, 128, 256]





        # ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        # env_gt = env_gt.view(3, 128, 256)  #+ ambient_gt

        # env_gt = env_gt * alpha
        # env_gt, alpha = self.tone(env_gt)

        assert self.label_nc == 6, 'wrong label_nc!!'

        env_gt = torch.cat([sg_map, sh_map])

        input_dict = {
            'input': env_gt,
            'crop': crop,
            # 'name': pair.replace('.png', ''),
            'name': pair.replace('.jpg', '').replace('.jfif',''),
            'warped': torch.zeros(3, 128, 256),
            'map': torch.zeros(1, 128, 256),
            'crop_path':crop_path
        }

        return input_dict

    def __len__(self):
        return self.dataset_size

##Test:SG+SH -->input:6
class TEST_SG_SH_fixedsh():

    def __init__(self, label_nc, SG_path, SH_path, CROP_path):

        self.SG_path = SG_path
        self.SH_path = SH_path
        self.label_nc = label_nc

        self.cropPath = CROP_path
        # self.pairs = os.listdir(CROP_path)
        self.pairs =  [img for img in os.listdir(CROP_path) if img.endswith('.png') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img]

        size = len(self.pairs)
        self.dataset_size = size
        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        self.to_tensor = transforms.ToTensor()

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):

        ln = 128
        pair = self.pairs[index]

        crop_path = osp.join(self.cropPath, pair)

        gt_path = osp.join(self.SG_path, pair.replace('.png', ''),
                           'sg_pred.pickle')
        sh_path = osp.join(self.SH_path, pair.replace('.png', ''),
                           'sh_pred.npy')

        #trial
        pair_string = pair.replace('.png', '')
        split_parts = pair_string.split('_')
        first_part = '_'.join(split_parts[:-4])
        # new_pair_string = first_part + "_0_"+'_'.join(split_parts[-3:])
        # new_pair_string = first_part + "_6_"+'_'.join(split_parts[-3:])
        new_pair_string = first_part + "_3_"+'_'.join(split_parts[-3:])

        # sh_path = osp.join(self.SH_path, 'AG8A9228_Panorama_hdr_Ref_view_ldr_0_110_expo_1.0',
        #                    'sh_pred.npy')#emlight
        # sh_path = osp.join(self.SH_path, 'AG8A9856_Panorama_hdr_Ref_view_ldr_6_70_expo_1.0',
        #                    'sh_pred.npy')#emlight
        sh_path = osp.join(self.SH_path, new_pair_string,
                           'sh_pred.npy')#emlight

        # if not os.path.exists(sh_path):
        #     sh_path = osp.join(self.SH_path, 'AG8A9856_Panorama_hdr_Ref_view_ldr_6_70_expo_1.0',
        #                        'sh_pred.npy')#emlight

        # if not os.path.exists(sh_path):
        #     sh_path = osp.join(self.SH_path, 'AG8A9856_Panorama_hdr_Ref_view_ldr_0_70_expo_1.0',
        #                        'sh_pred.npy')#emlight

        if not os.path.exists(sh_path):
            sh_path = osp.join(self.SH_path, 'AG8A9856_Panorama_hdr_Ref_view_ldr_1_70_expo_1.0',
                               'sh_pred.npy')#emlight


        # sh_path ="X:\\Users\\zhaoj1\\sg_roialign\\comparisions_tip\\em_salenet_sh_miao_retrain\\AG8A9856_Panorama_hdr_Ref_view_ldr_6_70_expo_1.0\\sh_pred.npy"

        # crop = util.load_exr(crop_path)
        # crop, alpha = self.tone(crop)
        # crop = cv2.resize(crop, (128, 128))
        # crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        crop = cv2.imread(crop_path)
        crop = cv2.resize(crop, (128, 128))
        crop = crop[..., ::-1]
        crop = self.to_tensor(crop.copy()).float().cuda()

        handle = open(gt_path, 'rb')
        pkl = pickle.load(handle)

        dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        intensity_gt = torch.from_numpy(np.array(
            pkl['intensity'])).float().cuda() * 0.01
        rgb_ratio_gt = torch.from_numpy(np.array(
            pkl['rgb_ratio'])).float().cuda()
        # ambient_gt = torch.from_numpy(
        #     pkl['ambient']).float().cuda() / (128 * 256)

        intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        dirs = util.sphere_points(ln)
        dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        size = torch.ones((1, ln)).cuda().float() * 0.0025
        light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        env_gt = util.convert_to_panorama(dirs, size, light_gt)

        # irradiance map
        sh = np.load(sh_path)  # (9,3)
        sh_map = sphericalHarmonics.shReconstructDiffuseMap(
            sh, width=256)  # (128,256,3)
        sh_map = np.transpose(sh_map, (2, 0, 1))  # (3, 128, 256)
        sh_map = torch.from_numpy(sh_map).float().cuda()

        # ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        env_gt = env_gt.view(3, 128, 256)  #+ ambient_gt

        # env_gt = env_gt * alpha
        # env_gt, alpha = self.tone(env_gt)

        assert self.label_nc == 6, 'wrong label_nc!!'

        env_gt = torch.cat([env_gt, sh_map])

        input_dict = {
            'input': env_gt,
            'crop': crop,
            'distribution': dist_gt,
            'intensity': intensity_gt,
            'name': pair.replace('.png', ''),
            'warped': torch.zeros(3, 128, 256),
            'map': torch.zeros(1, 128, 256)
        }

        return input_dict

    def __len__(self):
        return self.dataset_size
##Test:SG+SH -->input:6
class TEST_SG_SH_exr():

    def __init__(self, label_nc, SG_path, SH_path, CROP_path):

        self.SG_path = SG_path
        self.SH_path = SH_path
        self.label_nc = label_nc

        self.cropPath = CROP_path
        # self.pairs = os.listdir(CROP_path)
        # self.pairs =  [img for img in os.listdir(CROP_path) if img.endswith('.png') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img]
        self.pairs =  [img for img in os.listdir(CROP_path) if img.endswith('.exr') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img]

        size = len(self.pairs)
        self.dataset_size = size
        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        self.to_tensor = transforms.ToTensor()


        self.crop_tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        self.handle = util.PanoramaHandler()

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):

        ln = 128
        pair = self.pairs[index]

        crop_path = osp.join(self.cropPath, pair)

        gt_path = osp.join(self.SG_path, pair.replace('.exr', ''),
                           'sg_pred.pickle')
        sh_path = osp.join(self.SH_path, pair.replace('.exr', ''),
                           'sh_pred.npy')

        # crop = util.load_exr(crop_path)
        # crop, alpha = self.tone(crop)
        # crop = cv2.resize(crop, (128, 128))
        # crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        # crop = cv2.imread(crop_path)
        # crop = cv2.resize(crop, (128, 128))
        # crop = crop[..., ::-1]

        input = self.handle.read_hdr(crop_path)
        input,alpha = self.crop_tone(input)

        if input is None:
            print('Wrong path:', crop_path)
            exit(-1)
        elif input.shape[0] != 128 or input.shape[1] != 128:
            input = cv2.resize(input, dsize=(128,128))

        crop = self.to_tensor(input.copy()).float().cuda()

        handle = open(gt_path, 'rb')
        pkl = pickle.load(handle)

        dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        intensity_gt = torch.from_numpy(np.array(
            pkl['intensity'])).float().cuda() * 0.01
        rgb_ratio_gt = torch.from_numpy(np.array(
            pkl['rgb_ratio'])).float().cuda()
        ambient_gt = torch.from_numpy(
            pkl['ambient']).float().cuda() / (128 * 256)

        intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        dirs = util.sphere_points(ln)
        dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        size = torch.ones((1, ln)).cuda().float() * 0.0025
        light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        env_gt = util.convert_to_panorama(dirs, size, light_gt)

        # irradiance map
        sh = np.load(sh_path)  # (9,3)
        sh_map = sphericalHarmonics.shReconstructDiffuseMap(
            sh, width=256)  # (128,256,3)
        sh_map = np.transpose(sh_map, (2, 0, 1))  # (3, 128, 256)
        sh_map = torch.from_numpy(sh_map).float().cuda()

        ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        env_gt = env_gt.view(3, 128, 256)  #+ ambient_gt

        # env_gt = env_gt * alpha
        # env_gt, alpha = self.tone(env_gt)

        assert self.label_nc == 6, 'wrong label_nc!!'

        env_gt = torch.cat([env_gt, sh_map])

        input_dict = {
            'input': env_gt,
            'crop': crop,
            'distribution': dist_gt,
            'intensity': intensity_gt,
            'name': pair.replace('.exr', ''),
            'warped': torch.zeros(3, 128, 256),
            'map': torch.zeros(1, 128, 256)
        }

        return input_dict

    def __len__(self):
        return self.dataset_size


##Test:SG+SH -->input:6
class TEST_SG_SH_exr_interpolation():

    def __init__(self, label_nc, SG_path, SH_path, CROP_path,SG_path_gt,SH_path_gt,factors=[1.0]):

        self.SG_path = SG_path
        self.SH_path = SH_path
        self.SG_path_gt = SG_path_gt
        self.SH_path_gt = SH_path_gt
        self.factors=factors
        self.label_nc = label_nc

        self.cropPath = CROP_path
        # self.pairs = os.listdir(CROP_path)
        # self.pairs =  [img for img in os.listdir(CROP_path) if img.endswith('.png') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img]
        self.pairs =  [img for img in os.listdir(CROP_path) if img.endswith('.exr') and os.stat(osp.join(self.cropPath,img)).st_size>0 and '_view_' in img]

        size = len(self.pairs)
        self.dataset_size = size
        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        self.to_tensor = transforms.ToTensor()


        self.crop_tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        self.handle = util.PanoramaHandler()

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
            '.bmp', '.BMP', '.tiff', '.webp', '.exr'
        ]
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def __getitem__(self, index):

        ln = 128
        pair = self.pairs[index]

        crop_path = osp.join(self.cropPath, pair)

        sg_path = osp.join(self.SG_path, pair.replace('.exr', ''),
                           'sg_pred.pickle')
        sh_path = osp.join(self.SH_path, pair.replace('.exr', ''),
                           'sh_pred.npy')

        sg_path_gt = osp.join(self.SG_path_gt, pair.replace('.exr', '').replace('_view_ldr_','_ibl_').replace('_80','_SG.pickle'))
        sh_path_gt = osp.join(self.SH_path_gt, pair.replace('.exr', '').replace('_view_ldr_','_ibl_').replace('_80','_SH.npy'))


        # crop = util.load_exr(crop_path)
        # crop, alpha = self.tone(crop)
        # crop = cv2.resize(crop, (128, 128))
        # crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        # crop = cv2.imread(crop_path)
        # crop = cv2.resize(crop, (128, 128))
        # crop = crop[..., ::-1]

        input = self.handle.read_hdr(crop_path)
        input,alpha = self.crop_tone(input)

        if input is None:
            print('Wrong path:', crop_path)
            exit(-1)
        elif input.shape[0] != 128 or input.shape[1] != 128:
            input = cv2.resize(input, dsize=(128,128))

        crop = self.to_tensor(input.copy()).float().cuda()

        dirs = util.sphere_points(ln)
        dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        size = torch.ones((1, ln)).cuda().float() * 0.0025

        handle_sg = open(sg_path, 'rb')
        print("handle_sg:",handle_sg)
        pkl_src = pickle.load(handle_sg)

        handle_sg_gt = open(sg_path_gt, 'rb')
        print("handle_sg_gt:",handle_sg_gt)
        pkl_tgt = pickle.load(handle_sg_gt)


        iblCoeffs_src = np.load(sh_path)  # (9,3)
        iblCoeffs_tgt = np.load(sh_path_gt)  # (9,3)

        input_dicts=[]

        for i in range(len(self.factors)):
            factor = self.factors[i]
            pkl_inter={}

            pkl_inter['distribution'] = pkl_src['distribution'] * factor + pkl_tgt['distribution'] * (1-factor)
            pkl_inter['intensity'] = pkl_src['intensity'] * factor + pkl_tgt['intensity'] * (1-factor)
            pkl_inter['rgb_ratio'] = pkl_src['rgb_ratio']  * factor+ pkl_tgt['rgb_ratio']* (1-factor)
            iblCoeffs_inter= iblCoeffs_src * factor + iblCoeffs_tgt * (1-factor)    

            dist_gt = torch.from_numpy(pkl_inter['distribution']).float().cuda()
            intensity_gt = torch.from_numpy(np.array(
                pkl_inter['intensity'])).float().cuda() * 0.01
            rgb_ratio_gt = torch.from_numpy(np.array(
                pkl_inter['rgb_ratio'])).float().cuda()
            # ambient_gt = torch.from_numpy(
            #     pkl['ambient']).float().cuda() / (128 * 256)  

            intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
            dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
            rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)  
    

            light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
            env_gt = util.convert_to_panorama(dirs, size, light_gt) 

            # irradiance map
            
            sh_map = sphericalHarmonics.shReconstructDiffuseMap(
                iblCoeffs_inter, width=256)  # (128,256,3)
            sh_map = np.transpose(sh_map, (2, 0, 1))  # (3, 128, 256)
            sh_map = torch.from_numpy(sh_map).float().cuda()    

            # ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
            env_gt = env_gt.view(3, 128, 256)  #+ ambient_gt    

            # env_gt = env_gt * alpha
            # env_gt, alpha = self.tone(env_gt) 

            assert self.label_nc == 6, 'wrong label_nc!!'   

            env_gt = torch.cat([env_gt, sh_map])    

            input_dict = {
                'input': env_gt,
                'crop': crop,
                'distribution': dist_gt,
                'intensity': intensity_gt,
                'name': pair.replace('.exr', ''), #'name': pair.replace('.exr', '')+'_{}'.format(i),
                'warped': torch.zeros(3, 128, 256),
                'map': torch.zeros(1, 128, 256)
            }
            input_dicts.append(input_dict)

        return input_dicts

    def __len__(self):
        return self.dataset_size

def create_dataloader(opt):
    dataset = MyLavalIndoorDataset_SHSG(opt)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=not opt.serial_batches,
                                             num_workers=int(opt.nThreads),
                                             drop_last=opt.isTrain)
    return dataloader

def create_dataloader_fromexr(opt):
    dataset = MyLavalIndoorDataset_SHSG_fromexr(opt)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=not opt.serial_batches,
                                             num_workers=int(opt.nThreads),
                                             drop_last=opt.isTrain)
    return dataloader

def create_dataloader_fromexr_asg(opt):
    dataset = MyLavalIndoorDataset_SHSG_fromexr_asg(opt)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=not opt.serial_batches,
                                             num_workers=int(opt.nThreads),
                                             drop_last=opt.isTrain)
    return dataloader

def create_dataloader_TEST(opt, SG_path, SH_path,CROP_path):
    dataset = TEST_SG_SH(opt.label_nc, SG_path, SH_path, CROP_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=not opt.serial_batches,
                                             num_workers=int(opt.nThreads),
                                             drop_last=opt.isTrain)
    return dataloader

def create_dataloader_TEST_fromexr(opt, SG_path, SH_path,CROP_path):
    dataset = TEST_SG_SH_fromexr(opt.label_nc, SG_path, SH_path, CROP_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=int(opt.nThreads),
                                             drop_last=opt.isTrain)
    return dataloader


def create_dataloader_TEST_fromexr_SOTAcomparision(opt, SG_path, SH_path,CROP_path):
    dataset = TEST_SG_SH_fromexr_SOTAcomparision(opt.label_nc, SG_path, SH_path, CROP_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=int(opt.nThreads),
                                             drop_last=opt.isTrain)
    return dataloader

    

def create_dataloader_TEST_fixedsh(opt, SG_path, SH_path,CROP_path):
    dataset = TEST_SG_SH_fixedsh(opt.label_nc, SG_path, SH_path, CROP_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=not opt.serial_batches,
                                             num_workers=int(opt.nThreads),
                                             drop_last=opt.isTrain)
    return dataloader
def create_dataloader_TEST_exr(opt, SG_path, SH_path,CROP_path):
    dataset = TEST_SG_SH_exr(opt.label_nc, SG_path, SH_path, CROP_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=not opt.serial_batches,
                                             num_workers=int(opt.nThreads),
                                             drop_last=opt.isTrain)
    return dataloader

def create_dataloader_TEST_interpolation(opt, SG_path, SH_path,CROP_path,SG_path_gt,SH_path_gt,factors=[1.0]):
    dataset = TEST_SG_SH_exr_interpolation(opt.label_nc, SG_path, SH_path, CROP_path,SG_path_gt,SH_path_gt,factors)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=not opt.serial_batches,
                                             num_workers=int(opt.nThreads),
                                             drop_last=opt.isTrain)
    return dataloader