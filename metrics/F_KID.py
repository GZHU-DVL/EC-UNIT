"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy import linalg
from core.data_loader import get_eval_loader



try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3
        )
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x, SIFID=False):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)

def custom_kernel(X, Y=None, d=None):
    if Y is None:
        Y = X
    if d is None:
        d = X.shape[1]
    return ((1.0 / d) * np.dot(X, Y.T)) + 1

def mmd2(X, Y, kernel=custom_kernel, **kernel_args):
    K_XX = kernel(X, **kernel_args)
    K_XY = kernel(X, Y, **kernel_args)
    K_YY = kernel(Y, **kernel_args)
    return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

def calculate_kid(X, Y, n_subsets=100, subset_size=1000, kernel=custom_kernel, **kernel_args):
    mmds = []
    rng = np.random.RandomState(seed=0)
    subset_size = min(subset_size, len(X), len(Y))
    for _ in tqdm(range(n_subsets), desc="Calculating KID"):
        x_inds = rng.choice(len(X), subset_size, replace=False)
        y_inds = rng.choice(len(Y), subset_size, replace=False)
        X_subset = X[x_inds]
        Y_subset = Y[y_inds]
        mmds.append(mmd2(X_subset, Y_subset, kernel=kernel, **kernel_args))
    return np.mean(mmds), np.std(mmds)

@torch.no_grad()
def calculate_k_fid_given_paths(paths, img_size=256, batch_size=50):
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    load_fake=get_eval_loader(paths[1],
                              img_size,
                              batch_size,
                              is_count=True,
                              drop_last=True)
    load_real = get_eval_loader(root=paths[0],
                                img_size=img_size,
                                is_count=True,
                                batch_size=batch_size,
                                drop_last=True)
    loaders = [load_fake,load_real]

    mu, cov = [], []



    feature_list=[]
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, total=len(loader)):
            actv = inception(x.to(device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        feature_list.append(actvs)

        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))

    kid,_ = calculate_kid(feature_list[1], feature_list[0])
    fid = frechet_distance(mu[0], cov[0], mu[1], cov[1])


    return fid,kid


