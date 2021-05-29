#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : embedding_net.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-26 19:04
'''

import torch
import torch.nn as nn

class Embedding(nn.Module):

    def __init__(self, dim):
        super(Embedding, self).__init__()
        # https://pytorch.org/hub/pytorch_vision_googlenet/
        # https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py

        self.googlenet = torch.hub.load('pytorch/vision:v0.9.0', 'googlenet', pretrained=True)
        # print(*list(googlenet.children()))

        self.removed_fc_model = nn.Sequential(*list(self.googlenet.children())[:-1])
        # load
        # self.removed_fc_model.load_state_dict(googlenet)
        self.fc = nn.Linear(1024, dim)


    def forward(self, x):
        '''

        :param x: B, C, H, W = B, 3, 224, 224
        :return:
        '''

        x = self.removed_fc_model(x) # 1, 1024, 1, 1
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        x = self.fc(x) # 1, 400
        return x

if __name__ == '__main__':
    e = Embedding(400)
    x = torch.ones((8, 3, 300, 300))
    y = e(x)
    pass
