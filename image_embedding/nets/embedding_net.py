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
import torchvision.models as models


class Embedding(nn.Module):

    def __init__(self, dim, backbone_type="googlenet"):
        super(Embedding, self).__init__()
        assert backbone_type in ["googlenet", "resnet50"], "Only googlenet and resnet50 are supported!"
        self.backbone_type = backbone_type

        # https://pytorch.org/hub/pytorch_vision_googlenet/
        # https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py

        if self.backbone_type == "googlenet":
            # self.backbone = torchvision.load('pytorch/vision:v0.9.0', 'googlenet', pretrained=True)
            self.backbone = models.vgg19(pretrained=True, progress=True)
            # print(*list(googlenet.children()))
            self.removed_fc_model = nn.Sequential(*list(self.backbone.children())[:-1])
            self.fc = nn.Linear(1024, dim)
            # load
            # self.removed_fc_model.load_state_dict(googlenet)

        elif self.backbone_type == "resnet50":
            # self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
            self.backbone = models.resnet50(pretrained=True, progress=True)
            self.removed_fc_model = nn.Sequential(*list(self.backbone.children())[:-1])

            self.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(2048, dim)
            )

    def forward(self, x):
        '''

        :param x: B, C, H, W = B, 3, 224, 224
        :return:
        '''

        x = self.removed_fc_model(x)  # 1, 1024, 1, 1
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        x = self.fc(x)  # 1, 400
        return x


if __name__ == '__main__':
    e = Embedding(400, backbone_type="resnet50")
    x = torch.ones((8, 3, 300, 300))
    y = e(x)
    pass
