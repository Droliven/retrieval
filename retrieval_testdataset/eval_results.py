#!/usr/bin/env python
# encoding: utf-8
'''
@project : text_image_retrieval
@file    : eval_results.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-05-29 19:10
'''

import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from tqdm import tqdm


def eval_results(retrieval_txt_base, result_json_path, test_img_base):
    result_json_list = os.listdir(result_json_path)

    for j in tqdm(result_json_list):
        with open(os.path.join(result_json_path, j), 'r', encoding='utf-8') as jsonfile:
            json_string = json.load(jsonfile)

        id = json_string['txt_id']
        with open(os.path.join(retrieval_txt_base, id+".txt"), 'r', encoding='utf-8') as txt:
            content = txt.read()
        print(f"\n{id} >>> {content}")

        result = json_string["result"]

        plt.figure()
        # plt.text(20, 20, f"{id} >>> {content}")
        for idx in range(10):
            img = cv2.imread(os.path.join(test_img_base, result[idx] + ".jpg"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 5, idx+1)
            plt.imshow(img)

        plt.show()
        plt.close()
        pass

    pass

if __name__ == '__main__':
    result_json_path = os.path.join("../lfs/retrieval_cos_ascend")
    retrieval_txt_base = os.path.join("../lfs/sample_txt")
    test_img_base = os.path.join(r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\test\img")

    eval_results(retrieval_txt_base, result_json_path, test_img_base)

    pass