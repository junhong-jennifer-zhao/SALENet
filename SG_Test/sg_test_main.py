import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dataT
import numpy as np
from util import PanoramaHandler, TonemapHDR, tonemapping
from PIL import Image
import util
from Transformer import DETR, Salenet_SG
import imageio
import pickle5 as pickle
from progress.bar import Bar
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    batch_size = 1
    device = 'cuda:0'

    test_dir = r'../figures'

    hdr_test_dataset = dataT.ParameterTestDataset(test_dir)
    test_dataloader = DataLoader(hdr_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,num_workers=3)

    model = Salenet_SG().to(device)
    model = nn.DataParallel(model, device_ids=[0])

    load_weight = True
    if load_weight:
        model.load_state_dict(torch.load("./models/latest_net.pth"))
        print('load trained model')
    
    SummaryDir = './sg_output'
    if not os.path.exists(SummaryDir):
        os.makedirs(SummaryDir)

    model.eval()
    tone = util.TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.99)

    print(len(test_dataloader))

    with torch.no_grad():
        bar = Bar('Processing', max=len(test_dataloader))

        for i, para in enumerate(test_dataloader):
            input = para['crop'].cuda()
            print(input.shape)
            pred_8, pred_16, pred_32, pred_64, pred, out_8, out_16, out_32, out_64, out,features = model(input)

            for b in range(input.shape[0]):
                sv_dir = os.path.join(SummaryDir,para['name'][b])
                if not os.path.exists(sv_dir):
                    os.makedirs(sv_dir)

                pred_SG = {}
                pred_SG['distribution'] = pred['distribution'][b].cpu().numpy()
                pred_SG['intensity'] = pred['intensity'][b].cpu().numpy()
                pred_SG['rgb_ratio'] = pred['rgb_ratio'][b].cpu().numpy()              
                with open(os.path.join(sv_dir,"sg_pred.pickle"), 'wb') as h:
                    pickle.dump(pred_SG, h, protocol=pickle.HIGHEST_PROTOCOL)
                    
                ## visualization
                dist_pred = pred['distribution'][b]
                intensity_pred = pred['intensity'][b]
                rgb_ratio_pred = pred['rgb_ratio'][b]

                dist_pred = dist_pred.view(128 ,1)

                dirs = util.sphere_points(128)
                dirs = torch.from_numpy(dirs).float()
                dirs = dirs.view(1, 128 * 3).cuda()
                size = torch.ones((1, 128)).cuda().float() * 0.0025

                intensity_pred = intensity_pred.view(1, 1, 1).repeat(1, 128, 3) * 500
                dist_pred = dist_pred.view(1, 128, 1).repeat(1, 1, 3)
                rgb_ratio_pred = rgb_ratio_pred.view(1, 1, 3).repeat(1, 128, 1)

                light_pred = (dist_pred * intensity_pred * rgb_ratio_pred).view(1, 128 * 3)
                env_pred = util.convert_to_panorama(dirs, size, light_pred)
                env_pred = np.squeeze(env_pred[0].detach().cpu().numpy())
                env_pred = tone(env_pred)[0].transpose((1, 2, 0)).astype('float32')
     
                np.save(os.path.join(sv_dir,'pred-{}'.format(para['name'][b].replace('_','-'))),env_pred)

                img_pred = env_pred *255.0
                img_pred =  Image.fromarray(img_pred.astype('uint8'))
                img_pred.save(os.path.join(sv_dir,'pred-{}.jpg'.format(para['name'][b].replace('_','-'))))

            bar.next()

        bar.finish()      
       
       