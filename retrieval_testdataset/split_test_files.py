#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : split_test_files.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-29 17:06
'''
import os
import shutil
from tqdm import tqdm

def split_test_img(test_img_prefix, test_path, test_img_save_root):
    with open(test_path, 'r', encoding='utf-8') as f:
        test = f.read().splitlines()

    for t in tqdm(test):
        city, id = t.split("/")
        if not os.path.exists(os.path.join(test_img_save_root, city)):
            os.makedirs(os.path.join(test_img_save_root, city))

        shutil.copy(os.path.join(test_img_prefix, t + ".jpg"), os.path.join(test_img_save_root, t + ".jpg"))


def split_test_txt(test_txt_prefix, test_path, test_txt_save_root):
    with open(test_path, 'r', encoding='utf-8') as f:
        test = f.read().splitlines()

    for t in tqdm(test):
        city, id = t.split("/")
        if not os.path.exists(os.path.join(test_txt_save_root, city)):
            os.makedirs(os.path.join(test_txt_save_root, city))

        shutil.copy(os.path.join(test_txt_prefix, t + ".txt"), os.path.join(test_txt_save_root, t + ".txt"))


if __name__ == '__main__':
    test_path = "../image_embedding/datas/test_path.txt"
    test_img_prefix = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\img_resized_1M\cities_instagram"

    test_img_save_root = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\test\img"
    split_test_img(test_img_prefix, test_path, test_img_save_root)

    # test_path = "../image_embedding/datas/test_path.txt"
    # test_txt_prefix = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\captions_resized_1M\cities_instagram"
    #
    # test_txt_save_root = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\test\txt"
    # split_test_txt(test_txt_prefix, test_path, test_txt_save_root)
