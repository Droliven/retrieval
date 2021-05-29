#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : retrieval_online.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-29 19:55
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
import cv2
import json
import matplotlib.pyplot as plt
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


def save_dataset_doc_vector(w2v, doc, test_img_embed_base_dir, test_img_path, test_img_base, len_vec=400, tfidf_weighted=False):
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

    new_doc=""
    for char in doc:
        if char in WHITE_LIST:
            new_doc += char
    words_in=new_doc.split()
    words_in = [word.lower() for word in words_in]
    # Gensim simple_preproces instead tokenizer, 过滤掉太长或太短的token
    tokens = gensim.utils.simple_preprocess(" ".join(words_in))
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

    # TODO retrieval
    retrieval_txt_embed(test_img_embed_base_dir, test_img_path, test_img_base, embedding)


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


def retrieval_txt_embed(test_img_embed_base_dir, test_img_path, test_img_base, doc_embed):
    with open(test_img_path, 'r', encoding='utf-8') as f:
        test_img_list = f.read().splitlines()
        test_img_list = np.array(test_img_list)

    test_img_embed_all = np.load(test_img_embed_base_dir)

    cosdis = dist_cosine(doc_embed[np.newaxis, :], test_img_embed_all)
    cosdis_ascending = np.argsort(cosdis, axis=1)  # 1, 1500
    sort_ascend = cosdis_ascending[0]
    sort_ascend_img = list(test_img_list[sort_ascend])

    plt.figure()
    # plt.text(20, 20, f"{id} >>> {content}")
    for idx in range(10):
        img = cv2.imread(os.path.join(test_img_base, sort_ascend_img[idx] + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 5, idx + 1)
        plt.imshow(img)

    plt.show()
    plt.close()




if __name__ == '__main__':
    wv2 = load_pretrained_word2vec("../wordvec/ckpt1M/word2vec_model_instaCities1M.model")
    test_img_embed = r"../lfs/test_img_embed.npy"
    test_img_path = os.path.join(r"../image_embedding/datas/test_path.txt")  #
    test_img_base = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\test\img"
    doc = "car"
    while True:
        doc = input()
        if doc == "q":
            break
        save_dataset_doc_vector(wv2, doc, test_img_embed, test_img_path, test_img_base, )

