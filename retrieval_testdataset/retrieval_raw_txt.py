#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : retrieval_raw_txt.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-29 16:04
'''

import os
import os.path as osp
import gensim.downloader
from gensim.models import Word2Vec
from gensim.test.utils import datapath
import pprint
import string
from stop_words import get_stop_words
import numpy as np
from glob import glob
from tqdm import tqdm
import json


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
    if model_named is None:
        # google 预训练的模型
        w2v = gensim.downloader.load("word2vec-google-news-300")

        # Basic Usage
        # pprint.pprint(w2v.evaluate_word_analogies(datapath('questions-words.txt'), restrict_vocab=3000)[0])
        # print(w2v.most_similar("google"))
        # print(w2v.most_similar(positive=['america', 'newyork'], negative=['china'], topn=5))
        # print(w2v.similarity("beijing", "car"))
        # print(w2v.similarity("beijing", "london"))
        # print(w2v.similarity("london", "newyork"))
        # print(w2v.doesnt_match(['newyork', 'cananda','london','chicago']))
    else:
        # 我们自己在Insta数据集上训练的模型
        w2v = Word2Vec.load(model_named).wv

        # Basic Usage
        # print(model.wv.most_similar(positive=['america', 'newyork'], negative=['china'], topn=5))
        # print(model.wv.similarity("beijing", "car"))
        # print(model.wv.similarity("beijing", "london"))
        # print(model.wv.similarity("london", "newyork"))
        # print(model.wv.doesnt_match(['newyork', 'cananda', 'london', 'chicago']))

    return w2v


def save_dataset_doc_vector(w2v, sample_root, result_root, len_vec=400, tfidf_weighted=False):
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
    sample_list = os.listdir(sample_root)
    w2v_vocab = set(w2v.key_to_index.keys())

    for txt_file in tqdm(sample_list):
        file_id = txt_file.split(".")[0]

        with open(os.path.join(sample_root, txt_file), 'r', encoding='utf-8') as file:
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
            if not tfidf_weighted:
                c = 0
                for tok in tokens_filtered:
                    try:
                        embedding += w2v[tok]
                        c += 1
                    except:
                        continue
                if c > 0:
                    embedding /= c

            if tfidf_weighted:
                # vec = tfidf_dictionary.doc2bow(tokens_filtered)
                # vec_tfidf = tfidf_model[vec]
                # for tok in vec_tfidf:
                #     word_embedding = model[tfidf_dictionary[tok[0]]]
                #     embedding += word_embedding * tok[1]
                raise NotImplementedError("暂时没实现tf-idf weighted w2v!")

            # min/max 归一化
            embedding = embedding - min(embedding)
            if max(embedding) > 0:
                embedding = embedding / max(embedding)

            np.save(osp.join(result_root, file_id+".npy"), embedding)


def dist_cosine(x, y, eps=1e-6):
    """
    :param x: m x k array
    :param y: n x k array
    :return: m x n array
    """
    xx = np.sum(x ** 2, axis=1) ** 0.5
    x = x / (xx[:, np.newaxis] + eps)
    yy = np.sum(y ** 2, axis=1) ** 0.5
    y = y / (yy[:, np.newaxis] + eps)
    dist = 1 - np.dot(x, y.transpose())  # 1 - 余弦距离
    return dist


def retrieval_txt_embed(test_img_embed_base_dir, test_img_path, txt_embedding_base_dir, retrieval_cos_ascend):
    with open(test_img_path, 'r', encoding='utf-8') as f:
        test_img_list = f.read().splitlines()
        test_img_list = np.array(test_img_list)

    test_img_embed_all = np.load(test_img_embed_base_dir)
    txt_embedding_list = os.listdir(txt_embedding_base_dir)
    for txt in tqdm(txt_embedding_list):
        id = txt.split(".")[0]
        txt_embed = np.load(os.path.join(txt_embedding_base_dir, txt))
        cosdis = dist_cosine(txt_embed[np.newaxis, :], test_img_embed_all)
        cosdis_ascending = np.argsort(cosdis, axis=1)  # 1, 1500
        sort_ascend = cosdis_ascending[0]
        sort_ascend_img = test_img_list[sort_ascend]
        json_result = {'txt_id': id, 'result': list(sort_ascend_img)[:20]}
        with open(os.path.join(retrieval_cos_ascend, id + ".json"), 'w', encoding='utf-8') as f:
            json.dump(json_result, f)


if __name__ == '__main__':
    # wv2 = load_pretrained_word2vec("../wordvec/ckpt1M/word2vec_model_instaCities1M.model")
    # # txt_root = "/home/liuzhian/hdd4T/datasets/instaCities1M/captions_resized_1M/cities_instagram"
    # txt_root = r"../lfs/sample_txt"
    # embed_root = r"../lfs/retrieval_txt_embed"
    #
    # save_dataset_doc_vector(wv2, txt_root, embed_root)

    retrieval_txt_embed("../lfs/test_img_embed.npy", "../image_embedding/datas/test_path.txt", "../lfs/retrieval_txt_embed", "../lfs/retrieval_cos_ascend")
