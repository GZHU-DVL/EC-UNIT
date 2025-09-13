"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import re
import shutil
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch

from metrics.F_KID import calculate_k_fid_given_paths
from metrics.LPIPS import calculate_lpips_given_images
from core.data_loader import get_eval_loader
from core import utils

@torch.no_grad()
def eval(netEC, netG, args, step):
    
    if args.mode == 'viz':
        print('Generate images for visualization')
    elif args.mode == 'calc':
        print('Generate images for quantitative evaluation')
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    device = torch.device('cuda' if args.device=='cuda' else 'cpu')

    lpips_dict = OrderedDict()                      ## create a dictionary with order
    src_domain = args.src_domain;
    trg_domain = args.trg_domain;


    path_ref = args.target_paths
    loader_ref = get_eval_loader(root=path_ref,
                                    img_size=args.img_size,
                                    src_num=args.src_num,
                                    trg_num=args.trg_num,
                                    is_src=False,
                                    is_count=False,
                                    batch_size=args.val_batch_size,
                                    imagenet_normalize=False,
                                    drop_last=True,
                                    num_workers=args.numworkers)

    path_src = args.source_paths
    loader_src = get_eval_loader(root=path_src,
                                 img_size=args.img_size,
                                 src_num=args.src_num,
                                 trg_num=args.trg_num,
                                 is_src=True,
                                 is_count=False,
                                 batch_size=args.val_batch_size,
                                 imagenet_normalize=False,
                                 drop_last=True,
                                 num_workers=args.numworkers)
    name_Ec=re.split('/',args.content_encoder_path)[-1]
    name_G=re.split('/',args.generator_path)[-1]
    task = '%s2%s-use:%s_and_%s' % (src_domain, trg_domain,name_Ec, name_G)
    comptask='%s2%s_comp-use:%s_and_%s' % (src_domain, trg_domain,name_Ec, name_G)
    path_fake = os.path.join(args.eval_dir, task)
    path_comp=os.path.join(args.eval_dir,comptask)
    shutil.rmtree(path_fake, ignore_errors=True) 
    os.makedirs(path_fake)                       
    shutil.rmtree(path_comp, ignore_errors=True)  
    os.makedirs(path_comp)
    if args.mode=="calc":  
        lpips_values = []
        print('Generating images and calculating LPIPS for %s...' % task)
    else:  
        print("Generating images for visualization")
    for i, x_src in enumerate(tqdm(loader_src, total=len(loader_src))):
        N = x_src.size(0)                        ## N = batch
        x_src = x_src.to(device)## y_trg shape:[N,1] elements is trg_idx
       ## masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None ## i guess it is not important

        # generate 10 outputs from the same input
        group_of_images = []
        for j in range(args.num_outs_per_domain): ## default is 10
            try:
                x_ref = next(iter_ref).to(device)
            except:
                iter_ref = iter(loader_ref)   ##mode :reference,loader_ref = target
                x_ref = next(iter_ref).to(device)

            if x_ref.size(0) > N:
                x_ref = x_ref[:N]
            # print(x_src.shape,x_ref.shape);
            x_cfeat=netEC(x_src.to(device),get_feature=True)
            x_fake,_ = netG(x_cfeat, x_ref.to(device))
            group_of_images.append(x_fake)

            # save generated images to calculate FID later

            for k in range(N):
                filename = os.path.join(
                    path_fake,
                    '%.4i_%.2i.png' % (i*args.val_batch_size+(k+1), j+1))
                compname=os.path.join(
                    path_comp,
                    '%.4i_%.2i.png' % (i * args.val_batch_size + (k + 1), j + 1))
              #  print(torch.concat([x_src[k].unsqueeze(dim=0),x_ref[k].unsqueeze(dim=0),x_fake[k].unsqueeze(dim=0)],dim=0).shape)
                utils.save_image(x_fake[k],ncol=1,filename=filename);
                utils.save_image(torch.concat([x_src[k].unsqueeze(dim=0),x_ref[k].unsqueeze(dim=0),x_fake[k].unsqueeze(dim=0)],dim=0), ncol=1, filename=compname)

        if args.mode=="calc":
            lpips_value = calculate_lpips_given_images(group_of_images)
            lpips_values.append(lpips_value)

    # calculate LPIPS for each task (e.g. cat2dog, dog2cat)
    if args.mode=="calc":
        lpips_mean = np.array(lpips_values).mean()
        lpips_dict['LPIPS_%s' % (task)] = lpips_mean

    # delete dataloaders
    del loader_src
    # if mode == 'reference':
    del loader_ref
    del iter_ref

    # report LPIPS values
    if args.mode=="calc":
        filename = os.path.join(args.eval_dir, 'LPIPS-use:%s_and_%s.json' % (name_Ec,name_G))
        utils.save_json(lpips_dict, filename)

        # calculate and report fid values
        calculate_fid_for_all_tasks(args, trg_domain, src_domain,  name_Ec=name_Ec, name_G=name_G)


def calculate_fid_for_all_tasks(args, trg_domain, src_domain, name_Ec, name_G):

    fid_kid_values = OrderedDict()
    src_domain = src_domain
    trg_domain = trg_domain

    task = '%s2%s-use:%s_and_%s' % (src_domain, trg_domain,name_Ec,name_G)
    path_real = args.ref_img_paths
    path_fake = [os.path.join(args.eval_dir, task)]
    print('Calculating FID and KID for %s...' % task)
    fid, kid = calculate_k_fid_given_paths(
        paths=[path_real, path_fake],
        img_size=args.img_size,
        batch_size=args.val_batch_size)
    fid_kid_values['FID/%s' %  (task)] = fid
    fid_kid_values['KID/%s' % (task)] = kid


    # report FID values
    filename = os.path.join(args.eval_dir, 'FID_use:%s_and_%s.json' % (name_Ec,name_G))
    utils.save_json(fid_kid_values, filename)