#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : glove_utils.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-06-21 17:24
'''

import os
import os.path as osp
import gensim.downloader
from gensim.models import KeyedVectors
import string
from stop_words import get_stop_words
import numpy as np
from glob import glob
from tqdm import tqdm

WHITE_LIST = string.ascii_letters + string.digits + ' '
EN_STOP = get_stop_words("en")


def load_pretrained_word2vec(model_named=None):
    """
    加载预训练de word2vec
    :param model_named: 模型名/路径
    :type model_named: str
    :return:
    :rtype:
    """
    w2v = KeyedVectors.load_word2vec_format(os.path.join(model_named), binary=False)
    return w2v


def save_dataset_doc_vector(w2v, city_root_path, len_vec=256, fv_base_root=""):
    """
    基于训练好的word2vec模型，将一个doc中的所有词计算对应的特征向量，并将特征向量求平均作为doc的特征向量
    :param w2v: 训练好的模型
    :type w2v: gensim.Word2Vec.wv
    :param city_root_path: 每个城市的txt文件根目录夹
    :type city_root_path: str
    :param len_vec: 特征向量维度,默认400
    :type len_vec: int
    :param tfidf_weighted: 是否使用tf-idf weight
    :type tfidf_weighted: bool
    :param fv_base_root: 保存路径
    :type fv_base_root: str

    :return: 特征向量
    :rtype: np.ndarray
    """
    w2v_vocab = set(w2v.key_to_index.keys())

    # 创建好文件夹
    fv_city_root = osp.join(fv_base_root, "glove_feat")
    os.makedirs(fv_city_root, exist_ok=True)

    city_txt_files = glob(osp.join(city_root_path, "txt", "*.txt"))
    for txt_file in tqdm(city_txt_files):
        file_id = txt_file.split('/')[-1][:-4]
        with open(txt_file, 'r', encoding='utf-8') as file:
            caption = ""
            filtered_caption = ""

            for line in file:
                caption = caption + line

            # Replace hashtags with spaces
            caption = caption.replace('#', ' ')

            # Keep only letters and numbers
            for char in caption:
                if char in WHITE_LIST:
                    filtered_caption += char

            filtered_caption = filtered_caption.lower()
            # Gensim simple_preproces instead tokenizer, 过滤掉太长或太短的token
            tokens = gensim.utils.simple_preprocess(filtered_caption)
            stopped_tokens = [i for i in tokens if not i in EN_STOP]
            tokens_filtered = [token for token in stopped_tokens if token in w2v_vocab]

            embedding = np.zeros(len_vec)

            # mean of word2vec
            c = 0
            for tok in tokens_filtered:
                try:
                    embedding += w2v[tok]
                    c += 1
                except:
                    continue
            if c > 0:
                embedding /= c

            # # min/max 归一化
            # embedding = embedding - min(embedding)
            # if max(embedding) > 0:
            #     embedding = embedding / max(embedding)

            np.save(osp.join(fv_city_root, f"{file_id}.npy"), embedding)


if __name__ == '__main__':
    wv2 = load_pretrained_word2vec(r"./trained_glove/vectors.txt")
    # txt_root = r"E:\second_model_report_data\CN_insta_50K"
    txt_root = r"/mnt/hdd4T/dlw_home/model_report_data/datasets/CN_insta_50K"

    for city in ['beijing', 'shanghai', 'guangzhou', 'hangzhou']:
        save_dataset_doc_vector(wv2, city_root_path=osp.join(txt_root, city),
                                fv_base_root=os.path.join(txt_root, city), len_vec=256)

