#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : google_net.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-26 15:38
'''

import torch
import torch.nn as nn


# **************** images ****************
# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)


# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)  # wh = 1546, 1213
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# **************** Model ****************
# https://pytorch.org/hub/pytorch_vision_googlenet/
# https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py

googlenet = torch.hub.load('pytorch/vision:v0.9.0', 'googlenet', pretrained=True)
print(*list(googlenet.children()))


removed_fc_model = nn.Sequential(*list(googlenet.children())[:-2])


removed_fc_model.eval()
print(removed_fc_model)

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    removed_fc_model.to('cuda')

# for i, (name, parameters) in enumerate(model.named_parameters()):
#     print(i, name, ':', parameters.size())

with torch.no_grad():
    output = removed_fc_model(input_batch)  # 1, 1000

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)


