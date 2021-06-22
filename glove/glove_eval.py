#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : glove_eval.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-06-21 15:28
'''

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os
import pprint

# w2v = KeyedVectors.load_word2vec_format(os.path.join(r"E:\PythonWorkspace\multimodal_image_retrieval\text_image_retrieval\glove\glove.6B", "glove.6B.300d.txt"), binary=False)

'''
[('yahoo', 0.7205526828765869), ('microsoft', 0.6392883658409119), ('facebook', 0.6268066167831421), ('aol', 0.5989298820495605), ('internet', 0.5866513848304749), ('web', 0.5548399686813354), ('netscape', 0.5541645288467407), ('online', 0.5486863255500793), ('youtube', 0.5466190576553345), ('ebay', 0.5290301442146301)]
[('corresponsales', 0.36462923884391785), ('mayores', 0.35943520069122314), ('presbyterian', 0.33929523825645447), ('weill', 0.33522728085517883), ('croplife', 0.3343476951122284)]
[('china', 0.6774339079856873), ('chinese', 0.6219223141670227), ('hong', 0.5769438147544861), ('kong', 0.569959282875061), ('seoul', 0.5529952645301819)]
0.16640502
0.30604997
0.60757244
newyork
'''

# w2v = KeyedVectors.load_word2vec_format(os.path.join(r"E:\PythonWorkspace\multimodal_image_retrieval\text_image_retrieval\glove\glove.twitter.27B", "glove.twitter.27B.200d.txt"), binary=False)

'''
[('microsoft', 0.7242060303688049), ('facebook', 0.7069699764251709), ('maps', 0.7004954218864441), ('app', 0.6936156153678894), ('apple', 0.6872318983078003), ('yahoo', 0.6761190891265869), ('search', 0.6709437370300293), ('youtube', 0.6569339036941528), ('apps', 0.6430841684341431), ('web', 0.6382794380187988)]
[('nyc', 0.594921350479126), ('newyorkcity', 0.5608912706375122), ('chicago', 0.5443280339241028), ('newjersey', 0.5412189960479736), ('nashville', 0.5397787094116211)]
[('seoul', 0.6368500590324402), ('hongkong', 0.599950909614563), ('hong', 0.5892208218574524), ('tokyo', 0.5830789804458618), ('taiwan', 0.5611891746520996)]
0.19592683
0.5366771
0.4457513
cananda
'''

# w2v = KeyedVectors.load_word2vec_format(os.path.join(r"E:\PythonWorkspace\multimodal_image_retrieval\text_image_retrieval\glove\glove.840B.300d", "glove.840B.300d.txt"), binary=False)

'''
[('facebook', 0.7305588126182556), ('yahoo', 0.710445761680603), ('Google', 0.6869000792503357), ('youtube', 0.6741772890090942), ('firefox', 0.6372264623641968), ('twitter', 0.6232910752296448), ('wikipedia', 0.6119664311408997), ('adsense', 0.604214608669281), ('adwords', 0.6009767651557922), ('seo', 0.5990585088729858)]
[('york', 0.5794795155525208), ('newyorkcity', 0.5543256998062134), ('nyc', 0.5519964694976807), ('ny', 0.5301412343978882), ('manhattan', 0.5295382142066956)]
[('hong', 0.7126621603965759), ('kong', 0.7012479901313782), ('singapore', 0.6612123250961304), ('taiwan', 0.6277895569801331), ('bangkok', 0.6199225783348083)]
0.12319691
0.52560335
0.648153
cananda
'''

w2v = KeyedVectors.load_word2vec_format(os.path.join(r"E:\PythonWorkspace\multimodal_image_retrieval\text_image_retrieval\lfs", "vectors.txt"), binary=False)
# Basic Usage

print(w2v.most_similar(positive=['beijing', 'park'], negative=['guangzhou'], topn=5))
print(w2v.most_similar(positive=['shanghai', 'park'], negative=['hangzhou'], topn=5))

print(w2v.most_similar('beijing'))
print(w2v.most_similar('dog'))
print(w2v.most_similar('building'))
print(w2v.most_similar('coffee'))
print(w2v.most_similar('phone'))
print(w2v.most_similar('park'))

print(w2v.similarity("beijing", "car"))
print(w2v.similarity("beijing", "london"))
print(w2v.similarity("shanghai", "hangzhou"))
print(w2v.doesnt_match(['newyork', 'cananda', 'london', 'chicago']))

pass
