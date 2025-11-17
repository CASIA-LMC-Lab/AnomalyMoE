import torch
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from functools import partial
import math
from tqdm import tqdm
import warnings

import pickle

import collections
import os
from os.path import join
import io

import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import wget
from PIL import Image
string_classes = str

from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format

from torchvision import models
from torchvision import transforms as T
from torch.utils.tensorboard.summary import hparams


def calculate_hm_loss_for_expert(en_maps, de_maps, p=0.9, factor=0.1):

    cos_loss_func = torch.nn.CosineSimilarity(dim=1)
    batch_size = en_maps[0].shape[0]
    total_loss_per_sample = torch.zeros(batch_size, device=en_maps[0].device)
    
    for item in range(len(en_maps)):
        a_ = en_maps[item].detach()
        b_ = de_maps[item]
        

        with torch.no_grad():
            point_dist = 1 - cos_loss_func(a_, b_).unsqueeze(1)
            k = int(point_dist.numel() * (1 - p))
            if k == 0: k = 1
            thresh = torch.topk(point_dist.view(-1), k=k)[0][-1]


        loss_per_sample = (1 - cos_loss_func(a_, b_)).mean(dim=[1, 2])
        total_loss_per_sample += loss_per_sample

        inds = (point_dist < thresh).squeeze(1) 
        grad_mask = inds.unsqueeze(1).expand_as(b_)
        
        def grad_hook(grad):
            return torch.where(grad_mask, grad * factor, grad)

        if b_.requires_grad:
            b_.register_hook(grad_hook)
            

    return total_loss_per_sample / len(en_maps)


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter



from torch.optim.lr_scheduler import _LRScheduler


class WarmCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

