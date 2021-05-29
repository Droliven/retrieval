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

class Config():
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

        elif self.platform == "dlw":
            self.device = "cuda:0"
            self.num_workers = 4
            self.train_batch_size = 120
            self.valid_batch_size = 120
            self.test_batch_size = 120
            self.img_base_dir = r"/mnt/hdd4T/dlw_home/model_report_data/multi_modal_image_retrieval/InstaCities1M/img_resized_1M/cities_instagram"  # 10 cities * 1e5 imgs = 1M, size=300*300
            self.txt_embedding_base_dir = r"/mnt/hdd4T/dlw_home/model_report_data/multi_modal_image_retrieval/w2v_feat_vectors/cities_instagram"


        self.embedding_dim = 400

        self.ckpt_dir = r"./lfs/"

        self.train_img_path = r"../image_embedding/datas/train_img_path.txt"
        self.valid_img_path = r"../image_embedding/datas/valid_img_path.txt"
        self.test_img_path = r"../image_embedding/datas/test_img_path.txt"

        self.train_txt_embed_path = r"../image_embedding/datas/train_txt_embed_path.txt"
        self.valid_txt_embed_path = r"../image_embedding/datas/valid_txt_embed_path.txt"
        self.test_txt_embed_path = r"../image_embedding/datas/test_txt_embed_path.txt"

        self.lr = 1e-3
        self.lr_descent_rate = 0.1
        self.sgd_momentum = 0.9



        # self.lr_descent_every = 1e5
        # self.iterations = 5e5
        # self.eval_iters_evary = 1e4

        self.lr_descent_every = 15
        self.epochs = 75
        self.eval_iters_evary = 2

