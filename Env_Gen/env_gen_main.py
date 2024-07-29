"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from networks.pix2pix_model import Pix2PixModel
import util
from progress.bar import Bar
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

opt = TestOptions().parse()
opt.dataroot = './'
opt.batchSize = 1
opt.name = r'SGSH_warp'
opt.checkpoints_dir = r'./models'
opt.which_epoch = 'latest'
# opt.dataset_mode = 'lavalindoor'
opt.label_nc = int(6)
opt.semantic_nc = int(6)

CROP_path = r'../figures' 

SG_path = '../SG_Test/sg_output'

SH_path = '../SH_Test/sh_output'

result_dir = './EM_output'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

dataloader = data.create_dataloader_TEST(opt, SG_path, SH_path, CROP_path)

model = Pix2PixModel(opt)
model.eval()

bar = Bar('Processing', max=len(dataloader))

print(opt.batchSize, len(dataloader))
time_total = 0

for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= 0:
        # print("Here 1")
        start_time = time.time()
        generated = model(data_i, mode='inference')
        end_time = time.time()
        time_total += (end_time - start_time)
        nm = data_i['name']
        for b in range(generated.shape[0]):
            images = OrderedDict([('input', data_i['input'][b]),
                                  ('fake_image', generated[b]),
                                  ('im', data_i['crop'][b])])
            util.save_test_images(images, nm[b], result_dir)
        bar.next()
bar.finish()
print("total_time:",time_total)
