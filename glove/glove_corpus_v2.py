#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : glove_corpus_v2.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-06-21 17:33
'''

import os
import re
import pickle
import data_helper
import numpy as np

if __name__ == "__main__":
    stopwords = data_helper.stopwordslist("./data/stopwords.txt")  # 去掉停用词

    # 标签
    labelpath = "./data/REN/label.txt"
    lfile = open(labelpath, 'r', encoding='UTF-8')
    labels = lfile.read().split('\n')
    lfile.close()

    labels = labels[0:1487]
    labels = [la.split() for la in labels]
    labels = [la[1:] for la in labels]
    labels = [[int(np.floor(float(l) * 100)) for l in la] for la in labels]

    # 文本
    textpath = "./data/REN/data.txt"
    tfile = open(textpath, 'r', encoding='UTF-8')
    texts = tfile.read().split('<d>')
    tfile.close()

    texts=texts[1:]
    texts = [re.sub(r"<doc_[0-9]*>", "", text) for text in texts]
    texts = [re.sub(r"\n", "", text) for text in texts]
    texts = [re.sub(r"( )+", "", text) for text in texts]

    totaltexts=data_helper.remove_words(texts, [], 1)

    f = open('./data/REN/corpus.txt', 'w', encoding='UTF-8')
    totaltexts=[' '.join(totaltext) for totaltext in totaltexts]
    totaltexts=' '.join(totaltexts)
    f.write(totaltexts)
    f.close()