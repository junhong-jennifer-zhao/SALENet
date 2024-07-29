import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dataT
from torch.optim import lr_scheduler
import numpy as np
from util import PanoramaHandler, TonemapHDR, tonemapping
from PIL import Image
import util
import DLA_SK
from SphericalHarmonics import sphericalHarmonics
from progress.bar import Bar
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if __name__ == '__main__':
    h = PanoramaHandler()
    batch_size = 1
    tone = util.TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.99)

    test_dir = r'../figures'
    hdr_test_dataset = dataT.ParameterTestDataset(test_dir)
    test_dataloader = DataLoader(hdr_test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)


    model = DLA_SK.RegNet_SH().cuda()
    model = nn.DataParallel(model) 

    load_weight = True
    if load_weight:
        weight_dict = torch.load("./models/latest_net.pth")
        model.load_state_dict(weight_dict)
        print('load trained model')
    
    SummaryDir = './sh_output'
    if not os.path.exists(SummaryDir):
        os.makedirs(SummaryDir)


    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    print(len(test_dataloader))
    total_time = 0

    with torch.no_grad():
         bar = Bar('Processing', max=len(test_dataloader))

         for i, para in enumerate(test_dataloader):

            input = para['crop'].cuda()
            pred = model(input)
            ambient_pred = pred['ambient']

            for b in range(ambient_pred.shape[0]):

                sv_dir = os.path.join(SummaryDir,para['name'][b])
                if not os.path.exists(sv_dir):
                    os.makedirs(sv_dir)

                crop = np.squeeze(input[b].detach().cpu().numpy()).transpose((1, 2, 0)) * 255.0
                crop = Image.fromarray(crop.astype('uint8')).resize((256, 256))

                sh_pred = np.reshape(ambient_pred[b].detach().cpu().numpy(),(9,3))
                np.save(os.path.join(sv_dir,'sh_pred'), sh_pred)

                sh_pred = tone(sphericalHarmonics.shReconstructDiffuseMap(sh_pred,256))[0].astype('float32')

                np.save(os.path.join(sv_dir,'pred_diffuseMap'), sh_pred)

                im = sh_pred*255.0
                im_pred = Image.fromarray(im.astype('uint8'))
                im_pred.save(os.path.join(sv_dir,'pred_diffuseMap.jpg'))
            bar.next()

    bar.finish()