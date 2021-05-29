#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : data_split.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-26 19:10
'''


import os
import numpy as np

def split_imgs(base_dir):
    # base_dir = os.path.join(
    #     r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\img_resized_1M\cities_instagram")  # 10 cities * 1e5 imgs = 1M, size=300*300

    train_factor = 0.80
    valid_factor = 0.05
    test_factor = 0.15

    cities = os.listdir(base_dir)
    for ci, c in enumerate(cities):
        images = np.array(os.listdir(os.path.join(base_dir, c)))
        idx = np.arange(len(images))
        np.random.shuffle(idx)
        train_idx = idx[:int(len(images) * train_factor)]
        valid_idx = idx[int(len(images) * train_factor):int(len(images) * (train_factor + valid_factor))]
        test_idx = idx[int(len(images) * (train_factor + valid_factor)):]

        train_data = [os.path.join(base_dir, c, img) for img in images[train_idx]]
        with open("train_path.txt", "a+", encoding='utf-8') as f:
            f.write("\n".join(train_data))
            if ci != len(cities) - 1:
                f.write("\n")

        valid_data = [os.path.join(base_dir, c, img) for img in images[valid_idx]]
        with open("valid_path.txt", "a+", encoding='utf-8') as f:
            f.write("\n".join(valid_data))
            if ci != len(cities) - 1:
                f.write("\n")

        test_data = [os.path.join(base_dir, c, img) for img in images[test_idx]]
        with open("test_path.txt", "a+", encoding='utf-8') as f:
            f.write("\n".join(test_data))
            if ci != len(cities) - 1:
                f.write("\n")

def split_corresponding_txt_embedding(txt_embedding_base_dir):
    # txt_embedding_base_dir = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\w2v_feat_vectors\cities_instagram"

    for mode in ['train', 'valid', 'test']:
        with open(f"{mode}_img_path.txt", "r", encoding='utf-8') as f:
            img_path = f.read().splitlines()  # F:\model_report_data\multimodal_image_retrieval\InstaCities1M\img_resized_1M\cities_instagram\chicago\1490132097225301178.jpg

        txt_embed_path_line_list = []
        for l in img_path:
            city_id = l.split("\\")[-2:]
            txt_embed_path_line = os.path.join(txt_embedding_base_dir, city_id[0], city_id[1][:-4] + '.npy')
            txt_embed_path_line_list.append(txt_embed_path_line)

        with open(f"{mode}_txt_embed_path.txt", "a+", encoding='utf-8') as f:
            f.write("\n".join(txt_embed_path_line_list))

def split_caption_txt(caption_txt):
    # txt_embedding_base_dir = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\w2v_feat_vectors\cities_instagram"

    for mode in ['train', 'valid', 'test']:
        with open(os.path.join("../image_embedding/datas", f"{mode}_img_path.txt"), "r", encoding='utf-8') as f:
            img_path = f.read().splitlines()  # F:\model_report_data\multimodal_image_retrieval\InstaCities1M\img_resized_1M\cities_instagram\chicago\1490132097225301178.jpg

        caption_txt_line_list = []
        for l in img_path:
            city_id = l.split("\\")[-2:]
            caption_txt_line = os.path.join(caption_txt, city_id[0], city_id[1][:-4] + '.txt')
            caption_txt_line_list.append(caption_txt_line)

        with open(f"{mode}_caption_txt_path.txt", "a+", encoding='utf-8') as f:
            f.write("\n".join(caption_txt_line_list))


if __name__ == '__main__':
    for mode in ['train', 'valid', 'test']:
        with open(os.path.join(rf"C:\Users\Drolab\Desktop\{mode}_img_path.txt"), 'r', encoding='utf-8') as f:
            data = f.read().splitlines()

        data_city_id_list = []
        for d in data:
            city_id = d.split("/")[-2:]
            data_city_id_list.append("/".join(city_id)[:-4])

        with open(rf"{mode}_img_path.txt", 'a+', encoding='utf-8') as f:
            f.write("\n".join(data_city_id_list))
    pass














