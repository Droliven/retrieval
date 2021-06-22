#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : config.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-26 15:56
'''

import getpass


class Config:
    def __init__(self):
        self.platform = getpass.getuser()
        if self.platform == "Drolab":
            self.device = "cuda:0"
            self.num_workers = 0
            self.train_batch_size = 2
            self.valid_batch_size = 2
            self.test_batch_size = 2
            self.img_base_dir = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\img_resized_1M\cities_instagram"  # 10 cities * 1e5 imgs = 1M, size=300*300
            self.txt_embedding_base_dir = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\w2v_feat_vectors\cities_instagram"
            self.cos_ascend_base_dir = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\cos_ascend"

        elif self.platform == "liuzhian":
            self.device = "cuda:0"
            self.num_workers = 0
            self.train_batch_size = 2
            self.valid_batch_size = 2
            self.test_batch_size = 2
            self.ds_root = r"/home/liuzhian/hdd4T/datasets/CN_insta_50K"  # 10 cities * 1e5 imgs = 1M, size=300*300
            self.cos_ascend_base_dir = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\cos_ascend"

        elif self.platform == "dlw":
            self.device = "cuda:0"
            self.num_workers = 4
            self.train_batch_size = 128
            self.valid_batch_size = 128
            self.test_batch_size = 128
            self.ds_root="/mnt/hdd4T/dlw_home/model_report_data/datasets/CN_insta_50K"
            self.cos_ascend_base_dir = r"/mnt/hdd4T/dlw_home/model_report_data/multi_modal_image_retrieval/InstaCities1M/cos_ascend"

        self.backbone_type = "resnet50"
        self.embedding_dim = 256

        self.ckpt_dir = r"./lfs/"

        self.lr = 1e-3
        self.lr_descent_rate = 0.7
        self.sgd_momentum = 0.9

        # self.lr_descent_every = 1e5
        # self.iterations = 5e5
        # self.eval_iters_evary = 1e4

        self.lr_descent_every = 15
        self.epochs = 75
        self.eval_iters_evary = 1
