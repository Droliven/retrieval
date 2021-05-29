#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : img_embed.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-28 17:05
'''

from image_embedding.nets.embedding_net import Embedding
from image_embedding.datas.dataset import ImgDataset
from image_embedding.config import Config
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

class ImgEmbed():
    def __init__(self):
        self.cfg = Config()
        self.model = Embedding(self.cfg.embedding_dim)
        if self.cfg.device != "cpu":
            self.model = self.model.to(self.cfg.device)

        model_dict = self.model.state_dict()  # 688

        state = \
        torch.load("lfs/epoch75_currloss0.6658314065760876_bestloss0.6658251436359911.pth.tar", map_location=self.cfg.device)[
            "state_dict"]  # 344
        # print(state.keys())

        pretrained_dict = {k: v for k, v in state.items() if k in model_dict}

        model_dict.update(pretrained_dict)

        self.model.load_state_dict(model_dict)

        self.test_set = ImgDataset(self.cfg.img_base_dir, self.cfg.txt_embedding_base_dir, self.cfg.test_path)
        self.test_dataloader = DataLoader(self.test_set, batch_size=120, num_workers=self.cfg.num_workers, shuffle=False)
        print(f"test len: {self.test_set.__len__()}")

    def embed(self):
        # eval
        img_embed_npy = np.zeros((self.test_set.__len__(), 400))
        cusor = 0
        self.model.eval()
        for img, txt_emb in tqdm(self.test_dataloader):
            if self.cfg.device != "cpu":
                img = img.float().to(self.cfg.device)

            with torch.no_grad():
                out_emb = self.model(img)
                out_emb = out_emb.detach().cpu().data.numpy()

            img_embed_npy[cusor:cusor+img.shape[0], :] = out_emb
            cusor = cusor + img.shape[0]

        np.save(os.path.join(self.cfg.ckpt_dir, "test_img_embed.npy"), img_embed_npy)

if __name__ == "__main__":
    embed = ImgEmbed()
    embed.embed()

    # test_embed = np.load("./lfs/test_img_embed.npy")

    pass




