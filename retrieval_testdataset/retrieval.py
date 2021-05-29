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


def retrieval_from_seperate_txt_embed():
    with open(os.path.join("../image_embedding/datas/test_txt_embed_path.txt"), "r", encoding='utf-8') as f:
        txt_embed_path_list = f.read().splitlines()

    test_img_embed = np.load("./lfs/test_img_embed.npy")  # 150000, 400

    sort_ascend_npy = np.zeros((len(test_img_embed), len(test_img_embed)))

    for idx, txt_embed_path in enumerate(txt_embed_path_list):
        txt_embed = np.load(txt_embed_path)
        cosdis = dist_cosine(txt_embed[np.newaxis, :], test_img_embed)
        cosdis_ascending = np.argsort(cosdis,axis=1) # 1, 1500
        sort_ascend_npy[idx] = cosdis_ascending

    np.save("./txt_img_cos_ascend.npy", sort_ascend_npy)

def retrieval_from_allinone_txt_embed():
    test_txt_embed = np.load("./test_txt_embedding.npy")
    test_img_embed = np.load("./lfs/test_img_embed.npy")  # 150000, 400

    for idx in range(test_txt_embed.shape[0]):
        txt_embed = test_img_embed[idx]
        cosdis = dist_cosine(txt_embed[np.newaxis, :], test_img_embed)

        cosdis_ascending = np.argsort(cosdis,axis=1) # 1, 1500
        sort_ascend_npy = cosdis_ascending

        np.save(f"./lfs/retrieval/txt_img_cos_ascend_{idx}.npy", sort_ascend_npy)

retrieval_from_allinone_txt_embed()