#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : test.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-26 15:56
'''

# sample execution (requires torchvision)
import torch
from PIL import Image
from torchvision import transforms
import os
import numpy as np

# train_factor = 0.80
# valid_factor = 0.05
# test_factor = 0.15
#
# valid_data = ['I', 'am', 'droliven']
# b = "\n".join(valid_data)

from retrieval_testdataset.retrieval import dist_cosine
a = np.ones((400))
b = np.ones((10, 400))
d = dist_cosine(a[np.newaxis, :], b)
pass


