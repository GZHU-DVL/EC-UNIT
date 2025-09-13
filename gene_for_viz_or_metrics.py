"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import numpy as np
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
from metrics.eval  import eval as Eval
from model.content_encoder import ContentEncoder
from model.generator import Generator
# from othermodels.loader import load_checkpoint
import argparse


# cpu_limit_percent = 50
torch.set_num_threads(4)
class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="metric for Adversarial Image Translation of EC-UNIT")
        self.parser.add_argument("--numworkers", type = int, default=1,
                                help='the num of cpuworker')
        self.parser.add_argument("--task", type=str, default="",
                                 help='task description')
        self.parser.add_argument("--content_encoder_path", type=str, default='./checkpoint/content_encoder.pt',
                                 help="path to the saved content encoder")
        self.parser.add_argument("--generator_path", type=str, default='./checkpoint/cat2dog.pt',
                                 help="path to the generator of EC-UNIT")
        self.parser.add_argument("--val_batch_size", type=int, default=8,
                                 help="image size per batch for validation")
        self.parser.add_argument("--src_num",type=int,nargs='+',default=[9999999],
                                 help="the number of pictures from sources_domain")
        self.parser.add_argument("--trg_num",type=int,nargs='+',default=[9999999],
                                 help="the number of pictures from target_domain")
        self.parser.add_argument("--source_paths", type=str, nargs='+',
                                 help="the paths to source domain images")
        self.parser.add_argument("--target_paths", type=str, nargs='+',
                                 help="the paths to target domain images")

        self.parser.add_argument('--eval_dir', type=str, default='./eval',
                                 help='Directory for saving metrics, i.e., FID, KID, and LPIPS')
        self.parser.add_argument('--ref_img_paths',type=str,nargs='+',
                                 help="path to the reference images used as standards for " \
                                 "evaluation metrics that require reference data (e.g., FID, KID).")
        self.parser.add_argument('--img_size', type=int, default=256,
                                 help='Image resolution')
        self.parser.add_argument('--device', type=str, default='cuda',
                                 help="'cpu' for using cpu and 'GPU' for using GPU")
        self.parser.add_argument('--trg_domain', type=str, default='dog',
                                 help="The types of main subjects in target-domain images")
        self.parser.add_argument('--src_domain', type=str, default='cat',
                                 help="The types of main subjects in source-domain images")
        self.parser.add_argument('--num_outs_per_domain',type=int, default=10,
                                  help='Number of generated images per content input during sampling')
        self.parser.add_argument('--is_recovery',type=str, default='No',
                                 help='whether to recover the y_bar')
        self.parser.add_argument('--mode',type=str, default='viz',
                                 help='viz or calc, Generate images for visualization or calculate the metrics')
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

if __name__ == "__main__":
    parser = TrainOptions()
    args = parser.parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(torch.cuda.is_available())
    netEC=ContentEncoder()
    
    netG=Generator()
    
    
    netEC.load_state_dict(torch.load(args.content_encoder_path, map_location=lambda storage, loc:storage))
    ckpt = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
    netG.load_state_dict(ckpt['g_ema'])
    netEC.eval()
    netG.eval()
    netG = netG.to(device)
    netEC=netEC.to(device) 
    print('Load models successfully')

    Eval(netEC,netG,args,0)
   


