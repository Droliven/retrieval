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
import os.path as osp
from tqdm import tqdm
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImgDataset(Dataset):
    def __init__(self, ds_root, split="train"):
        super(ImgDataset, self).__init__()

        self.ds_root = ds_root
        assert split in ["train", "val", "test"]
        self.split = split
        self.cities = ["beijing", "shanghai", "guangzhou", "hangzhou"]

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.sample_ids = []

        for city in self.cities:
            with open(osp.join(ds_root, city, "%s.txt" % split), "r", encoding='utf-8') as f:
                city_ids = f.read().splitlines()
                # 添加城市前缀
                city_ids = [osp.join(city, id) for id in city_ids]

                self.sample_ids.extend(city_ids)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # print(f"{item}-{self.img_path[item]}")
        sample_id = self.sample_ids[idx]
        city = sample_id.split("/")[0]

        img_path = osp.join(self.ds_root, sample_id + ".jpg")
        img_path = img_path.replace(city, osp.join(city, "img")) # 加上img文件夹前缀

        txt_feat_path = osp.join(self.ds_root, sample_id + ".npy")
        # # w2v 数据
        # txt_feat_path = txt_feat_path.replace(city, osp.join(city, "w2v_feat"))
        # glove 数据
        txt_feat_path = txt_feat_path.replace(city, osp.join(city, "glove_feat"))

        img = Image.open(img_path).convert('RGB')  # wh = 1546, 1213
        if len(np.array(img).shape) == 2:
            img = np.stack([np.array(img), ] * 3, axis=-1)
            img = Image.fromarray(img)

        img = self.preprocess(img)
        txt_embed = np.load(txt_feat_path)
        return img, txt_embed


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dset = ImgDataset(ds_root=r"/home/liuzhian/hdd4T/datasets/CN_insta_50K")
    dl = DataLoader(dset, batch_size=2)

    for idx, (img, txt_emb) in tqdm(enumerate(dl)):
        print(idx)
        print(img)
        print(txt_emb)
        pass
