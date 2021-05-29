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

import torch
from image_embedding.nets.embedding_net import Embedding

model = Embedding(400)
model_dict = model.state_dict()  # 688
print(model_dict.keys())

state = torch.load("lfs/epoch75_currloss0.6658314065760876_bestloss0.6658251436359911.pth.tar", map_location="cuda:0")["state_dict"]  # 344
print(state.keys())

pretrained_dict = {k: v for k, v in state.items() if k in model_dict}

model_dict.update(pretrained_dict)

model.load_state_dict(model_dict)


pass


