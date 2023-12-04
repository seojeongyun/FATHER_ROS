#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# The code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py
import math
from copy import deepcopy
import torch
import torch.nn as nn


class ModelEMA_arc:
    def __init__(self, model, decay=0.9999, updates=0):
        self.in_features = model.in_features
        self.out_features = model.out_features
        self.s = model.s
        self.m = model.m
        self.weight = deepcopy(model.weight.detach())
        #
        self.updates = updates
        self.decay_init = decay
        #
        self.weight.requires_grad_(False)

    def decay(self, x):
        return self.decay_init * (1 - math.exp(-x / 2000))

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            decay = self.decay(self.updates)

            curr_weight = model.weight.detach()
            if curr_weight.dtype.is_floating_point:
                self.weight *= decay
                self.weight += (1 - decay) * curr_weight

    def load_weights(self, pretrained):
        self.in_features = pretrained.in_features
        self.out_features = pretrained.out_features
        self.m = pretrained.m
        self.s = pretrained.s
        self.weight = pretrained.weight


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from one instance and set them to another instance."""
    for k, item in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, item)
