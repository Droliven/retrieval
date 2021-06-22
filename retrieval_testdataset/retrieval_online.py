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
import string
from stop_words import get_stop_words
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from image_embedding.datas.dataset import ImgDataset
from sklearn.metrics.pairwise import cosine_similarity

WHITE_LIST = string.ascii_letters + string.digits + ' '
EN_STOP = get_stop_words("en")

ds_root = "/home/liuzhian/hdd4T/datasets/CN_insta_50K"


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


def get_most_similar_imgs(w2v, doc, test_imgs_embed, len_vec=256, sample_ids=None,
                          tfidf_weighted=False):
    """
    基于训练好的word2vec模型，将一个doc中的所有词计算对应的特征向量，并将特征向量求平均作为doc的特征向量

    然后用这个doc去检索所有test set zhong de tu pian
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

    new_doc = ""
    for char in doc:
        if char in WHITE_LIST:
            new_doc += char
    words_in = new_doc.split()
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
    retrieval_txt_embed(test_imgs_embed, embedding, sample_ids)


def retrieval_txt_embed(test_imgs_embed, doc_embed, sample_ids, top_n=20):
    cosdis = cosine_similarity(doc_embed[np.newaxis, :], test_imgs_embed)

    most_similar_idxs = np.argsort(cosdis, axis=1)[0, :top_n]
    most_similar_sample_ids = [sample_ids[idx] for idx in most_similar_idxs]
    print(str(most_similar_sample_ids))

    imgs_res = []
    for sample_id in most_similar_sample_ids:
        city = sample_id.split("/")[0]
        img_path = osp.join(ds_root, sample_id + ".jpg")
        img_path = img_path.replace(city, osp.join(city, "img"))  # 加上img文件夹前缀

        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        imgs_res.append(img)

    grid_vis = []
    # 每行展示5张图片
    img_vis = np.array(imgs_res)
    for row_id in range(len(img_vis )// 5):
        vis_row = img_vis[row_id * 5:(row_id + 1) * 5, :, :, :]
        vis_row = np.hstack(vis_row)
        grid_vis.append(vis_row)

    grid_vis = np.vstack(grid_vis)
    cv2.imshow("res", grid_vis)
    cv2.waitKey(0)


if __name__ == '__main__':
    w2v = load_pretrained_word2vec("../wordvec/ckpt/word2vec_model_CNInsta50K.model")
    test_img_embed = np.load(r"../lfs/test_img_embed.npy")
    ds_test = ImgDataset(ds_root="/home/liuzhian/hdd4T/datasets/CN_insta_50K", split="test")

    while True:
        docs = ['university','towel','girl','boy','bicycle']
        for doc in docs:
            print(doc)
            get_most_similar_imgs(w2v, doc, test_img_embed, 256, ds_test.sample_ids)
