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


def save_dataset_doc_vector(w2v, txt_files_list, len_vec=400, tfidf_weighted=False,
                            fv_base_root="/home/liuzhian/hdd4T/datasets/instaCities1M/w2v_feat_vectors/cities_instagram"):
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
    os.makedirs(fv_base_root, exist_ok=True)

    all_embeddings = np.zeros((len(test_caption_txt_path_list), len_vec))

    for idx, (txt_file) in tqdm(enumerate(txt_files_list)):
        file_id = txt_file.split('\\')[-1][:-4]
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
            all_embeddings[idx] = embedding

    np.save(osp.join(fv_base_root, "test_txt_embedding.npy"), all_embeddings)


if __name__ == '__main__':
    wv2 = load_pretrained_word2vec("../wordvec/ckpt1M/word2vec_model_instaCities1M.model")

    with open(os.path.join("../image_embedding/datas/test_caption_txt_path.txt"), "r", encoding='utf-8') as f:
        test_caption_txt_path_list = f.read().splitlines()

    save_dataset_doc_vector(wv2, txt_files_list=test_caption_txt_path_list,
                                fv_base_root=r"./")
