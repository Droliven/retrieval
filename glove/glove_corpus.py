#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : glove_corpus.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-06-21 17:28
'''
import os.path

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import gensim
import string
import glob
from tqdm import tqdm


whitelist = string.ascii_letters + string.digits + ' '
words2filter = ['rt', 'http', 't', 'gt', 'co', 's', 'https', 'http', 'tweet', 'markars_', 'photo', 'pictur', 'picture',
                'say', 'photo', 'much', 'tweet', 'now', 'blog', 'wikipedia', 'google', 'flickr', 'figure', 'photo',
                'image', 'homepage', 'url', 'youtube', 'wikipedia', 'google', 'flickr', 'figure', 'photo', 'image',
                'homepage', 'url', 'youtube', 'images', 'blog', 'pinterest']

# # create English stop words list
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)


class InstaCities1MCorpus:
    def __init__(self, instagram_text_data_path, cities):
        self.instagram_text_data_path = instagram_text_data_path
        self.cities = cities

    def __iter__(self):
        # -- LOAD DATA FROM INSTAGRAM --
        for city in self.cities:
            print("Loading InstaCities1M data from " + city)
            for i, file_name in tqdm(enumerate(glob.glob(os.path.join(self.instagram_text_data_path, city, "txt", "*.txt")))):
                caption = ""
                filtered_caption = ""
                file = open(file_name, "r", encoding='utf-8')
                for line in file:
                    caption = caption + line
                # Replace hashtags with spaces
                caption = caption.replace('#', ' ')
                # Keep only letters and numbers
                for char in caption:
                    if char in whitelist:
                        filtered_caption += char

                # to lower
                filtered_caption = filtered_caption.lower()

                # Gensim simple_preproces instead tokenizer, 把太长或太短的单词忽略掉
                # 每次产生一个list[str]，每个元素是一个token
                tokens = gensim.utils.simple_preprocess(filtered_caption)
                # remove stop words from tokens
                stopped_tokens = [i for i in tokens if not i in en_stop]
                yield stopped_tokens

if __name__ == '__main__':
    import os.path as osp
    instagram_text_data_path = osp.join(r"/mnt/hdd4T/dlw_home/model_report_data/datasets/CN_insta_50K")
    # instagram_text_data_path = osp.join(r"E:\second_model_report_data\CN_insta_50K")
    cities = ["beijing", "shanghai", "guangzhou", "hangzhou"]
    # cities = ["shanghai"]
    insta_cities_1m_corpus = InstaCities1MCorpus(instagram_text_data_path, cities)
    tokens = []
    for i, data in tqdm(enumerate(insta_cities_1m_corpus)):
        tokens += data

    with open(osp.join(r"../lfs/", "glove_corpus.txt"), "w", encoding='utf-8') as f:
        f.write(" ".join(tokens))

