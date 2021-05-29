#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : dataset.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-26 19:21
'''
from tqdm import tqdm
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np

class ImgDataset(Dataset):
    def __init__(self, img_path="", txt_embed_path=""):
        super(ImgDataset, self).__init__()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with open(img_path, "r", encoding='utf-8') as f:
            self.img_path = f.read().splitlines()

        with open(txt_embed_path, "r", encoding='utf-8') as f:
            self.txt_embed_path = f.read().splitlines()

        assert len(self.txt_embed_path) == len(self.txt_embed_path)


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        # print(f"{item}-{self.img_path[item]}")

        img = Image.open(self.img_path[item]).convert('RGB')  # wh = 1546, 1213
        if len(np.array(img).shape) == 2:
            img = np.stack([np.array(img), ] * 3, axis=-1)
            img = Image.fromarray(img)

        img = self.preprocess(img)
        txt_embed = np.load(self.txt_embed_path[item])
        return img, txt_embed

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dset = ImgDataset("./valid_img_path.txt", "./valid_txt_embed_path.txt")
    dl = DataLoader(dset, batch_size=2)

    for idx,(img, txt_emb, path) in tqdm(enumerate(dl)):
        # print(idx)
        # print(img)
        # print(txt_emb)
        pass

