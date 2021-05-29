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
    def __init__(self, img_path_prefix=r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\img_resized_1M\cities_instagram", txt_embed_prefix="", path=r"./test_path.txt"):
        super(ImgDataset, self).__init__()

        self.img_path_prefix = img_path_prefix
        self.txt_embed_prefix = txt_embed_prefix

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with open(path, "r", encoding='utf-8') as f:
            self.path = f.read().splitlines()


    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        # print(f"{item}-{self.img_path[item]}")

        img = Image.open(os.path.join(self.img_path_prefix, self.path[item] + '.jpg')).convert('RGB')  # wh = 1546, 1213
        if len(np.array(img).shape) == 2:
            img = np.stack([np.array(img), ] * 3, axis=-1)
            img = Image.fromarray(img)

        img = self.preprocess(img)
        txt_embed = np.load(os.path.join(self.txt_embed_prefix, self.path[item] + '.npy'))
        return img, txt_embed


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dset = ImgDataset()
    dl = DataLoader(dset, batch_size=2)

    for idx,(img, txt_emb) in tqdm(enumerate(dl)):
        # print(idx)
        # print(img)
        # print(txt_emb)
        pass

