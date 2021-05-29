from image_embedding.config import Config
from retrieval_testdataset.img_embed import ImgEmbed
from retrieval_testdataset.retrieval_raw_txt import load_pretrained_word2vec, save_dataset_doc_vector, retrieval_txt_embed

# # 对测试集所有图片编码
# embed = ImgEmbed()
# embed.embed()

# # 对测试文本编码
# wv2 = load_pretrained_word2vec("./wordvec/ckpt1M/word2vec_model_instaCities1M.model")
# txt_root = r"./lfs/sample_txt"
# embed_root = r"./lfs/retrieval_txt_embed"
# save_dataset_doc_vector(wv2, txt_root, embed_root)
# 检索
retrieval_txt_embed("../lfs/test_img_embed.npy", "../image_embedding/datas/test_path.txt", "../lfs/retrieval_txt_embed",
                    "../lfs/retrieval_cos_ascend")

# 检索测试集文本与图片特征的相似度
# cfg = Config()
# test_path = cfg.test_path
# txt_embedding_base_dir = cfg.txt_embedding_base_dir
#
# cos_ascend_base_dir = r"F:\model_report_data\multimodal_image_retrieval\InstaCities1M\cos_ascend"
# retrieval_txt_embed(test_path, txt_embedding_base_dir, cos_ascend_base_dir)
