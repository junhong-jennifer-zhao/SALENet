"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.networks.base_network import BaseNetwork
from networks.networks.normalization import get_nonspade_norm_layer
from networks.networks.architecture import ResnetBlock as ResnetBlock
from networks.networks.architecture import SPADEResnetBlock as SPADEResnetBlock,Adapter,ConvAdapter

from .spherenet import SphereConv2D, SphereMaxPool2D


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt, is_adapter_enc=False, is_adapter_dec=False):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        self.up = nn.Upsample(scale_factor=2)

        self.sphere_conv1 = SphereConv2D(nf, 3, stride=1)
        self.sphere_pool1 = SphereMaxPool2D(stride=2)


        self.is_adapter_enc = is_adapter_enc
        self.is_adapter_dec = is_adapter_dec
        self.netE = ConvEncoder(opt,self.is_adapter_enc)

        if self.is_adapter_dec:
            self.head_0_adapter = ConvAdapter(16 * nf,16 * nf)
            self.G_middle_adapter = ConvAdapter(16 * nf,16 * nf)
            self.up_0_adapter = ConvAdapter(16 * nf,8 * nf)
            self.up_1_adapter = ConvAdapter(8 * nf,4 * nf)
            self.up_2_adapter = ConvAdapter(4 * nf,2 * nf)
            self.up_3_adapter = ConvAdapter(2 * nf,1 * nf)
            self.end_adapter = ConvAdapter(1 * nf,3)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers) # 2**5 = 32, 64, 128.
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, crop):
        guide = input
        batch_size = input.shape[0]
        x = self.netE(crop)
        x = x.view(-1, 16 * self.opt.ngf, 1, 2)
        x = F.interpolate(x, size=(self.sh, self.sw))

        x_input = x
        x = self.head_0(x, guide)
        if self.is_adapter_dec:
            head_0_adapter_output = self.head_0_adapter(x_input)
            x_res = head_0_adapter_output
            x = x + x_res

        x = self.up(x)
        x_input = x

        x = self.G_middle_0(x, guide)

        x = self.G_middle_1(x, guide)
        x_input_shape = x.shape

        if self.is_adapter_dec:
            G_middle_adapter_output = self.G_middle_adapter(x_input)
            x_res = G_middle_adapter_output
            x = x + x_res


        x = self.up(x)
        x_input = x
        x = self.up_0(x, guide)
        x_input_shape = x.shape

        if self.is_adapter_dec:
            up_0_adapter_output = self.up_0_adapter(x_input)
            x_res = up_0_adapter_output
            x = x + x_res

        x = self.up(x)
        x_input = x

        x = self.up_1(x, guide)
        x_input_shape = x.shape

        if self.is_adapter_dec:
            up_1_adapter_output = self.up_1_adapter(x_input)
            x_res = up_1_adapter_output
            x = x + x_res

        x = self.up(x)
        x_input = x

        x = self.up_2(x, guide)
        x_input_shape = x.shape

        if self.is_adapter_dec:
            up_2_adapter_output = self.up_2_adapter(x_input)
            x_res = up_2_adapter_output
            x = x + x_res

        x = self.up(x)
        x_input = x

        x = self.up_3(x, guide)#[6, 512, 128, 256]
        x_input_shape = x.shape



        if self.is_adapter_dec:
            up_3_adapter_output = self.up_3_adapter(x_input)
            x_res = up_3_adapter_output
            x = x + x_res

        x_input = x
        x = self.sphere_conv1(F.leaky_relu(x, 2e-1))

        if self.is_adapter_dec:
            end_adapter_output = self.end_adapter(x_input)
            x_res = end_adapter_output
            x = x + x_res

        x = (F.tanh(x) + 1) * 25 * 0.23


        return x

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt, is_adapter=False):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf  # 64
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        # if opt.crop_size >= 256:
        # self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.fc = nn.Linear(ndf * 8 * 4 * 4, 16 * ndf * 2 * 1)    # 128, 64, 32, 16, 8, 4
        # self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256) 16 * ndf * 4, 8

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt
        self.is_adapter=is_adapter
        if is_adapter:
            self.layer1_adapter = ConvAdapter(3,1 * ndf,stride=2)
            self.layer2_adapter = ConvAdapter(1 * ndf,2 * ndf,stride=2)
            self.layer3_adapter = ConvAdapter(2 * ndf,4 * ndf,stride=2)
            self.layer4_adapter = ConvAdapter(4 * ndf,8 * ndf,stride=2)
            self.layer5_adapter = ConvAdapter(8 * ndf,8 * ndf,stride=2)
            self.fc_adapter = Adapter(ndf * 8 * 4 * 4, 16 * ndf * 2 * 1)


    def forward(self, x):
        # if x.size(2) != 256 or x.size(3) != 256:
        x = F.interpolate(x, size=(128, 128), mode='bilinear')

        x_input = x
        x = self.layer1(x)
        if self.is_adapter:
            layer1_adapter_output = self.layer1_adapter(x_input)
            x_res = layer1_adapter_output
            x = x + x_res
        
        x_input = x
        x = self.layer2(self.actvn(x))
        if self.is_adapter:
            layer2_adapter_output = self.layer2_adapter(x_input)
            x_res = layer2_adapter_output
            x = x + x_res

        x_input = x
        x = self.layer3(self.actvn(x))
        if self.is_adapter:
            layer3_adapter_output = self.layer3_adapter(x_input)
            x_res = layer3_adapter_output
            x = x + x_res


        x_input = x
        x = self.layer4(self.actvn(x))
        if self.is_adapter:
            layer4_adapter_output = self.layer4_adapter(x_input)
            x_res = layer4_adapter_output
            x = x + x_res

        x_input = x
        x = self.layer5(self.actvn(x))

        if self.is_adapter:
            layer5_adapter_output = self.layer5_adapter(x_input)
            x_res = layer5_adapter_output
            x = x + x_res

        x = x.view(x.size(0), -1)

        x_input = x
        x = self.actvn(x)
        z = self.fc(x)
        if self.is_adapter:
            fc_adapter_output = self.fc_adapter(x_input)
            z_res = fc_adapter_output
            z = z + z_res
        return z

