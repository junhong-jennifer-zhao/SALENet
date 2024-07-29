import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import random
import os
import util_pano
from util_pano import tonemapping_xjp
import cv2
from util_pano import PanoramaHandler
import time
import sys
# import pickle5 as pickle
import pickle
from progress.bar import Bar



class EnvMap_BRDF_renderLoss(Module):
    def __init__(self,imSize=256,exrWidth=256, sampleWidth=16,sample_times=1):
        super(EnvMap_BRDF_renderLoss, self).__init__()

        self.pre_path = '/media/common/xjp/Illumination-Estimation-main/dataset/brdf_128/pre_1_1250'

        self.l1 = nn.L1Loss(reduction='sum').cuda()
        self.l2 = nn.MSELoss(reduction='sum').cuda()

        self.nms = os.listdir('/media/common/xjp/Illumination-Estimation-main/dataset/brdf_128/pre_1_1250')
        
        self.model_num = len(self.nms)

        self.sampleWidth = sampleWidth 
        self.sampleHeight = sampleWidth//2

        self.exrWidth = exrWidth
        self.exrHeight = exrWidth//2

        self.imSize = imSize

        self.fov = 60 / 180.0 * np.pi
        self.F0 = 0.05
        self.cameraPos = np.array([0, 0, 0],
                                  dtype=np.float32).reshape([1, 3, 1, 1])
        self.yRange = self.xRange = 1 * np.tan(self.fov / 2)

        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, self.imSize),
                           np.linspace(-self.yRange, self.yRange, self.imSize))
        y = np.flip(y, axis=0)
        z = -np.ones((self.imSize, self.imSize), dtype=np.float32)

        pCoord = np.stack([x, y, z]).astype(np.float32)
        self.pCoord = pCoord[np.newaxis, :, :, :]
        v = self.cameraPos - self.pCoord
        v = v / np.sqrt(
            np.maximum(np.sum(v * v, axis=1), 1e-12)[:, np.newaxis, :, :])
        v = v.astype(dtype=np.float32)

        self.v = torch.from_numpy(v)
        self.pCoord = torch.from_numpy(self.pCoord)
        self.up = torch.Tensor([0, 1, 0])

        self.sample_times = sample_times

        ls = np.zeros((self.sample_times,self.sampleWidth*self.sampleHeight,3))
        envWeight = np.zeros((self.sample_times,self.sampleWidth*self.sampleHeight,1))

        for i in range(self.sample_times):
            Az = ((np.arange(self.sampleWidth) +  np.around(0.1*(-1)**i*np.floor((i+1)/2)+0.5, 1)) / self.sampleWidth - 0.5) * 2 * np.pi
            El = ((np.arange(self.sampleHeight) + np.around(0.1*(-1)**i*np.floor((i+1)/2)+0.5, 1)) / self.sampleHeight) * np.pi / 2.0
            Az, El = np.meshgrid(Az, El)
            Az = Az.reshape(-1, 1)
            El = El.reshape(-1, 1)
            lx = np.sin(El) * np.cos(Az)
            ly = np.sin(El) * np.sin(Az)
            lz = np.cos(El)
            ls[i] = np.concatenate((lx, ly, lz), axis=1)  #[sample_width*height,3]
            envWeight[i] = np.sin(El) * np.pi * np.pi / self.sampleWidth / self.sampleHeight

        self.ls = torch.from_numpy(ls.astype(np.float32))
        self.envWeight = torch.from_numpy(envWeight.astype(np.float32))
        self.envWeight = self.envWeight.unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1)

    def init_model(self, idx):
        self.nm = self.nms[idx]
        diffuse_nm = os.path.join(
            '/media/common/xjp/Illumination-Estimation-main/dataset/brdf_128/albedo',
            self.nm+'_albedo.npy')
        normal_nm = os.path.join(
            '/media/common/xjp/Illumination-Estimation-main/dataset/brdf_128/normal',
            self.nm+'_normal.npy')
        rough_nm = os.path.join(
            '/media/common/xjp/Illumination-Estimation-main/dataset/brdf_128/rough',
            self.nm+'_rough.npy')
        seg_nm = os.path.join(
            '/media/common/xjp/Illumination-Estimation-main/dataset/brdf_128/seg',
            self.nm+'_seg.npy')

        diffuse = np.load(diffuse_nm)
        normal = np.load(normal_nm)
        rough = np.load(rough_nm)
        seg = np.load(seg_nm)

        self.pixel_num = np.sum(seg)

        self.diffuse = np.expand_dims(diffuse, 0)
        self.diffuse = torch.from_numpy(self.diffuse)

        self.normal = np.expand_dims(normal, 0)
        self.normal = torch.from_numpy(self.normal)

        self.rough = np.expand_dims(rough, 0)
        self.rough = torch.from_numpy(self.rough)

        self.seg = np.expand_dims(seg, 0)
        self.seg = torch.from_numpy(self.seg)

        self.convert_to_gpu()
        
        self.idx_ = np.load(os.path.join(self.pre_path,self.nm,'idx_0.npy'))[np.newaxis,...]
        self.idy_ = np.load(os.path.join(self.pre_path,self.nm,'idy_0.npy'))[np.newaxis,...]

    def convert_to_gpu(self):
        if self.device != None:
            self.diffuse = self.diffuse.to(self.device)
            self.normal = self.normal.to(self.device)
            self.rough = self.rough.to(self.device)
            self.seg = self.seg.to(self.device)
            self.v = self.v.to(self.device)
            self.pCoord = self.pCoord.to(self.device)
            self.up = self.up.to(self.device)
            self.ls = self.ls.to(self.device)
            self.envWeight = self.envWeight.to(self.device)
        else:
            print('tensors are supposed to be in gpus')
            sys.exit(0)

    def light_intensity(self, environment_exr):
        return environment_exr[:,self.idy_[:],self.idx_[:],:].permute(0,1,5,2,3,4)

    def env_render(self, environment_exr):
        ldirections = self.ls.unsqueeze(-1).unsqueeze(-1)
        camyProj = torch.einsum('b,abcd->acd',
                                (self.up, self.normal)).unsqueeze(1).expand_as(
                                    self.normal) * self.normal

        camy = F.normalize(self.up.unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1).expand_as(camyProj) - camyProj,
                           dim=1)
        camx = -F.normalize(torch.cross(camy, self.normal, dim=1), p=1, dim=1)

        l = ldirections[ :,:, 0:1, :, :] * camx.unsqueeze(1) \
                + ldirections[ :,:, 1:2, :, :] * camy.unsqueeze(1) \
                + ldirections[ :,:, 2:3, :, :] * self.normal.unsqueeze(1) #[self.sample_times, sample_width*height, 3, 256, 256]

        h = (self.v.unsqueeze(1) + l) / 2
        h = h / torch.sqrt(
            torch.clamp(torch.sum(h * h, dim=2),
                        min=1e-6).unsqueeze(2))  #[1, sample_width*height, 3, 256, 256]

        vdh = torch.sum((self.v * h),
                        dim=2).unsqueeze(2)  #[sample_times, sample_width*height, 1, 256, 256]

        temp = torch.FloatTensor(1, 1, 1, 1, 1)

        if self.device != None:
            temp = temp.to(self.device)
        else:
            temp = temp.cuda()

        temp.data[0] = 2.0

        frac0 = self.F0 + (1 - self.F0) * torch.pow(
            temp.expand_as(vdh),
            (-5.55472 * vdh - 6.98316) * vdh)  #[sample_times, sample_width*height, 1, 256, 256]

        diffuseBatch = (self.diffuse + 1) / 2.0 / np.pi # [1, 3, 256, 256]
        roughBatch = (self.rough + 1.0) / 2.0

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        ndv = torch.clamp(
            torch.sum(self.normal * self.v.expand_as(self.normal), dim=1), 0,
            1).unsqueeze(1).unsqueeze(2)
        ndh = torch.clamp(torch.sum(self.normal.unsqueeze(1) * h, dim=2), 0,
                          1).unsqueeze(2)
        ndl = torch.clamp(torch.sum(self.normal.unsqueeze(1) * l, dim=2), 0,
                          1).unsqueeze(2)

        frac = alpha2.unsqueeze(1).expand_as(frac0) * frac0
        nom0 = ndh * ndh * (alpha2.unsqueeze(1).expand_as(ndh) - 1) + 1
        nom1 = ndv * (1 - k.unsqueeze(1).expand_as(ndh)) + k.unsqueeze(
            1).expand_as(ndh)
        nom2 = ndl * (1 - k.unsqueeze(1).expand_as(ndh)) + k.unsqueeze(
            1).expand_as(ndh)
        nom = torch.clamp(4 * np.pi * nom0 * nom0 * nom1 * nom2, 1e-6,
                          4 * np.pi)
        specPred = frac / nom  #[sample_times, sample_width*height, 1, 256, 256]
                            
        l=l.permute(0,2,1,3,4)  #[sample_times,3,sample_width*height,256,256]

        envmap = self.light_intensity(environment_exr) #[bn,sample_times,3,sample_width*height,256,256]

        envmap = envmap.permute(0,1,3,2,4,5) # [bn, sample_times,sample_width*height, 3, 256, 256]

        brdfDiffuse = diffuseBatch.unsqueeze(0).unsqueeze(0).expand([self.batch_size,self.sample_times ,self.sampleWidth*self.sampleHeight, 3, self.imSize, self.imSize] ) * \
                    ndl.unsqueeze(0).expand([self.batch_size,self.sample_times  ,self.sampleWidth*self.sampleHeight, 3,self.imSize, self.imSize] )   #[bn,self.sample_times, 128, 3, 256, 256]
        colorDiffuse = torch.sum(brdfDiffuse * envmap * self.envWeight.expand_as(brdfDiffuse), dim=(1,2))/self.sample_times

        brdfSpec = specPred.unsqueeze(0).expand([self.batch_size,self.sample_times, self.sampleWidth*self.sampleHeight, 3, self.imSize, self.imSize] ) * \
            ndl.unsqueeze(0).expand([self.batch_size,self.sample_times,self.sampleWidth*self.sampleHeight , 3, self.imSize, self.imSize] )
        colorSpec = torch.sum(brdfSpec * envmap * self.envWeight.expand_as(brdfSpec), dim=(1,2))/self.sample_times

        return  colorDiffuse + 10 * colorSpec

    def forward(self, x, y):

        self.batch_size = x.shape[0]
        self.device = x.device

        idx = random.randint(0, self.model_num - 1)

        self.init_model(idx)

        pred = self.env_render(x)
        gt = self.env_render(y)
        
        pixel_num = (self.pixel_num * self.batch_size * 3)
        loss = self.l2(pred, gt)/ pixel_num

        return loss, pred[0], gt[0]

class Pre_EnvMap_BRDF_renderloss(Module):
    def __init__(self,mode=0):
        super(Pre_EnvMap_BRDF_renderloss, self).__init__()
        self.l2 = nn.MSELoss(reduction='sum').cuda()

        if mode == 0:
            self.prePath = '/media/common/xjp/Illumination-Estimation-main/dataset/brdf_128/pre_2_1225'
            seg_path = '/media/common/xjp/Illumination-Estimation-main/dataset/brdf_128/seg'
            self.imSize = 128
        elif mode == 1:
            self.prePath = '/media/common/xjp/Illumination-Estimation-main/dataset/brdf/pre_2_1225'
            seg_path = '/media/common/xjp/Illumination-Estimation-main/dataset/brdf/seg'
            self.imSize = 256
        else:
            print("uncorrect mode!!!")
            sys.exit(0)
        self.mode = mode

        self.nms = os.listdir(self.prePath)
        self.lengh = len(self.nms)
       
        pixels = {}
        for nm in self.nms:
            seg_nm = nm+'_seg.npy'
            seg = np.load(os.path.join(seg_path, seg_nm))
            pixels[nm]=np.sum(seg)
        self.pixels = pixels

        self.range_times = 2

    def env_render(self, environment_exr, index):
        # environment_exr [bn,128,256,3]
        model_path = os.path.join(self.prePath, self.nms[index])
        renderings = torch.zeros(self.batch_size, 3, self.imSize,self.imSize).to(self.device)
        for i in range(self.range_times):
            # 1024, 1, 256, 256
            brdfSpec_i = torch.from_numpy(
                np.load(os.path.join(
                    model_path, 'brdfSpec_{}.npy'.format(i)))).to(self.device)
            # 1024, 3, 256, 256
            brdfDiffuse_i = torch.from_numpy(
                np.load(
                    os.path.join(model_path,
                                 'brdfDiffuse_{}.npy'.format(i)))).to(
                                     self.device)
            # 1024, 256, 256
            idx_i = np.load(os.path.join(
                model_path, 'idx_{}.npy'.format(i))).astype(np.int)
            idy_i = np.load(os.path.join(
                model_path, 'idy_{}.npy'.format(i))).astype(np.int)
            # 1,1024,1,1,1
            weight_i = torch.from_numpy(
                np.load(os.path.join(
                    model_path,
                    'weight_{}.npy'.format(i))).astype(np.float)).to(
                        self.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            # environment_exr[:,idy[:],idx[:],:].permute(0,1,4,2,3)  #bn, 1024, 3, 256, 256
            renderings = renderings + torch.sum(brdfDiffuse_i.unsqueeze(0)*weight_i*(environment_exr[:,idy_i[:],idx_i[:],:].permute(0,1,4,2,3)),dim=1)\
                +10* torch.sum(brdfSpec_i.unsqueeze(0)*weight_i*(environment_exr[:,idy_i[:],idx_i[:],:].permute(0,1,4,2,3)),dim=1)
        return renderings

    def forward(self, x, y):

        self.batch_size = x.shape[0]

        self.device = x.device

        index = random.randint(0, self.lengh-1)

        pred = self.env_render(x, index)
        gt = self.env_render(y, index)

        pixel_num = (self.pixels[self.nms[index]] * self.batch_size * 3)
        loss = self.l2(pred, gt) / pixel_num

        return loss, pred[0], gt[0]

class Pre_EnvMap_BRDF_renderloss_pkl(Module):
    def __init__(self,mode=0):
        super(Pre_EnvMap_BRDF_renderloss_pkl, self).__init__()
        self.l2 = nn.MSELoss(reduction='sum').cuda()
        self.l1 = nn.L1Loss(reduction='sum').cuda()

        if mode == 0:
            self.prePath = '/media/common/xjp/Illumination-Estimation-main/dataset/brdf_128/pre_2_1225'
            seg_path = '/media/common/xjp/Illumination-Estimation-main/dataset/brdf_128/seg'
            self.imSize = 128
        elif mode == 1:
            self.prePath = '/media/common/xjp/Illumination-Estimation-main/dataset/brdf/pre_2_1225'
            seg_path = '/media/common/xjp/Illumination-Estimation-main/dataset/brdf/seg'
            self.imSize = 256
        else:
            print("uncorrect mode!!!")
            sys.exit(0)
        self.mode = mode

        self.nms = os.listdir(self.prePath)
       
        pixels = {}
        for nm in self.nms:
            seg_nm = nm+'_seg.npy'
            seg = np.load(os.path.join(seg_path, seg_nm))
            pixels[nm]=np.sum(seg)
        self.pixels = pixels

        self.range_times = 2
        self.lengh = len(self.nms)

        self.init_pre()

    def init_pre(self):
        pre_dict={}
        pkl_path = '/media/common/xjp/Illumination-Estimation-main/dataset/brdf_128/pkl_2_1225'
        starttime = time.time()

        for i,nm in enumerate(self.nms): 
            handle = open(os.path.join(pkl_path,'{}.pickle'.format(nm)), 'rb')
            pre_dict[nm] = pickle.load(handle)
        self.pre_dict = pre_dict
        endtime = time.time()
        dtime = endtime - starttime


    def env_render(self, environment_exr, index,flag):
        preBrdf_dict = self.pre_dict[self.nms[index]]

        for i in range(self.range_times):

            brdfSpec_i = preBrdf_dict['brdfSpec_{}'.format(i)].to(self.device)

            brdfDiffuse_i = preBrdf_dict['brdfDiffuse_{}'.format(i)].to(self.device)

            idx_i = preBrdf_dict['idx_{}'.format(i)].astype(np.int)
            idy_i = preBrdf_dict['idy_{}'.format(i)].astype(np.int)

            weight_i = torch.from_numpy(preBrdf_dict['weight_{}'.format(i)]).to(self.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

            
            diffuse = torch.sum(brdfDiffuse_i.unsqueeze(0)*weight_i*(environment_exr[:,idy_i[:],idx_i[:],:].permute(0,1,4,2,3)),dim=1)
            spec = 10* torch.sum(brdfSpec_i.unsqueeze(0)*weight_i*(environment_exr[:,idy_i[:],idx_i[:],:].permute(0,1,4,2,3)),dim=1)                    
            
            if flag ==0 : 
                self.pred = self.pred + diffuse + spec
            else:
                self.gt = self.gt + diffuse + spec


    def forward(self, x, y):
        
        self.batch_size = x.shape[0]
        self.device = x.device

        self.pred = torch.zeros(self.batch_size, 3, self.imSize,self.imSize).to(self.device)
        self.gt = torch.zeros(self.batch_size, 3, self.imSize,self.imSize).to(self.device)

        index = random.randint(0, self.lengh-1)

        self.env_render(x, index,flag= 0)
        self.env_render(y, index,flag= 1)


        pixel_num = (self.pixels[self.nms[index]] * self.batch_size * 3)
        loss = self.l2(self.pred, self.gt) / pixel_num

        return loss, self.pred[0], self.gt[0]
