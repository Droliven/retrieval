#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : retrieval.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-28 22:22
'''
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from image_embedding.config import Config
from tqdm import tqdm

def dist_cosine(x, y):
    """
    :param x: m x k array
    :param y: n x k array
    :return: m x n array
    """
    xx = np.sum(x ** 2, axis=1) ** 0.5
    x = x / xx[:, np.newaxis]
    yy = np.sum(y ** 2, axis=1) ** 0.5
    y = y / yy[:, np.newaxis]
    dist = 1 - np.dot(x, y.transpose())  # 1 - 余弦距离
    return dist


def retrieval_txt_embed(test_path_dir, txt_embedding_base_dir, cos_ascend_base_dir):
    test_img_embed_all = np.load("./lfs/test_img_embed.npy")
    with open(test_path_dir, 'r', encoding='utf-8') as f:
        path = f.read().splitlines()

    for p in tqdm(path):
        city = p.split("/")[0]
        if not os.path.exists(os.path.join(cos_ascend_base_dir, city)):
            os.makedirs(os.path.join(cos_ascend_base_dir, city))

        txt_embed = np.load(os.path.join(txt_embedding_base_dir, p + '.npy'))
        cosdis = dist_cosine(txt_embed[np.newaxis, :], test_img_embed_all)
        cosdis_ascending = np.argsort(cosdis, axis=1)  # 1, 1500
        sort_ascend_npy = cosdis_ascending

        np.save(os.path.join(cos_ascend_base_dir, p + ".npy"), sort_ascend_npy)



if __name__ == '__main__':
    cfg = Config()
    test_path = cfg.test_path
    txt_embedding_base_dir = cfg.txt_embedding_base_dir

    cos_ascend_base_dir = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\cos_ascend"
    retrieval_txt_embed(test_path, txt_embedding_base_dir, cos_ascend_base_dir)
