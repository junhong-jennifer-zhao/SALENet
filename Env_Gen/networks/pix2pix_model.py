"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import networks.networks as networks
import util
import torch.nn.functional as F
from loss import Pre_EnvMap_BRDF_renderloss_pkl,Pre_EnvMap_BRDF_renderloss,EnvMap_BRDF_renderLoss

import time

class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt,is_pretrain=False):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD = self.initialize_networks(opt)

        self.is_pretrain = is_pretrain
        if self.is_pretrain:
            pretext_model = torch.load("/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_CMIC_02/Final_PART/checkpoints/lavalindoor_SGSH_warp_zcy/170_net_G_retrain.pth")
            model_dict = self.netG.state_dict()
            state_dict_backbone = {k:v for k,v in pretext_model.items() if (k in model_dict.keys() and 'adapter' not in k) }
            model_dict.update(state_dict_backbone)
            self.netG.load_state_dict(model_dict)
        self.netG = self.netG.cuda()
        for name, param in self.netG.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False


        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode,
                                                 tensor=self.FloatTensor,
                                                 opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()

            
            ############### TODO render loss ################
            if opt.use_renderloss:
                self.renderLoss = EnvMap_BRDF_renderLoss(imSize=128,exrWidth=256,sampleWidth=50,sample_times=1).cuda()
            #################################################
            print("opt.no_vgg_loss:",opt.no_vgg_loss)
            if not opt.no_vgg_loss:
                print("Hello!!!")
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input, crop, real_image, map = data['input'].cuda(), data['crop'].cuda(
        ), data['warped'].cuda(), data['map'].cuda()
        print("crop shape:",crop.shape)

        if mode == 'generator':
            
            if self.opt.use_renderloss:
                ##########TODO renderloss #############
                g_loss, generated, renderings = self.compute_generator_loss(
                    input, crop, real_image, map)
                return g_loss, generated, renderings
                #######################################

            g_loss, generated = self.compute_generator_loss(
                input, crop, real_image, map)
            return g_loss, generated

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input, crop, real_image)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_image = self.generate_fake(input, crop)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())

        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD

    def compute_generator_loss(self, input, crop, real_image, map):

        G_losses = {}
        fake_image = self.generate_fake(input, crop)
        cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        L2 = torch.nn.MSELoss()

        pred_fake, pred_real = self.discriminate(input, fake_image, real_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake,
                                            True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(
                        num_intermediate_outputs):  # for each layer output
                    # print(pred_fake[i][j].shape)
                    # print(1 / 0)
                    _, _, h, w = pred_fake[i][j].shape
                    map = F.interpolate(map, size=(h, w))
                    pred_fake_weighted = pred_fake[i][j] * map + pred_fake[i][
                        j] * (1 - map) * 50
                    pred_real_weighted = pred_real[i][j] * map + pred_real[i][
                        j] * (1 - map) * 50

                    weighted_loss = self.criterionFeat(
                        pred_fake_weighted, pred_real_weighted.detach())
                    GAN_Feat_loss += weighted_loss * 1 / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * 5
        G_losses['COS'] = (1 - cos_sim(fake_image, real_image)).mean() * 5


        if self.opt.use_renderloss:


            results = self.renderLoss(fake_image.permute(0,2,3,1),real_image.permute(0,2,3,1))
        
            G_losses['RenderLoss'] = results[0] * 100

            renderings =  [results[1],results[2]]

            return G_losses, fake_image, renderings

            #################################################

        return G_losses, fake_image

    def compute_discriminator_loss(self, input, crop, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image = self.generate_fake(input, crop)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(input, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake,
                                               False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real,
                                               True,
                                               for_discriminator=True)

        return D_losses

    def generate_fake(self, input, crop):
        fake_image = self.netG(input, crop)
        return fake_image

    def discriminate(self, input, fake_image, real_image):
        fake_concat = torch.cat([input, fake_image], dim=1)
        print("real_image shape:",real_image.shape)
        print("input shape:",input.shape)
        real_concat = torch.cat([input, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :,
             1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] !=
                                                   t[:, :, :, :-1])
        edge[:, :,
             1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] !=
                                                   t[:, :, :-1, :])
        return edge.float()

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
