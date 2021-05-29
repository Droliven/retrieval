"""训练word2vec"""
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import gensim
import string
import glob
import multiprocessing
import json
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

cores = multiprocessing.cpu_count()

finetune = False
if finetune:
    pretrained_model_path = './datasets/WebVision/models/word2vec/word2vec_model_webvision.model'
    model = gensim.models.Word2Vec.load(pretrained_model_path)

whitelist = string.ascii_letters + string.digits + ' '
instagram_text_data_path = "/home/liuzhian/hdd4T/datasets/instaCities1M/captions_resized_1M/cities_instagram"
webvision_text_data_path = './datasets/WebVision/'
model_path = 'ckpt1M/word2vec_model_instaCities1M.model'
words2filter = ['rt', 'http', 't', 'gt', 'co', 's', 'https', 'http', 'tweet', 'markars_', 'photo', 'pictur', 'picture',
                'say', 'photo', 'much', 'tweet', 'now', 'blog', 'wikipedia', 'google', 'flickr', 'figure', 'photo',
                'image', 'homepage', 'url', 'youtube', 'wikipedia', 'google', 'flickr', 'figure', 'photo', 'image',
                'homepage', 'url', 'youtube', 'images', 'blog', 'pinterest']
cities = ['chicago', 'london', 'losangeles', 'newyork', 'sanfrancisco', 'sydney', 'singapore', 'melbourne',
          'miami', 'toronto']

size = 400  # vector size
min_count = 5  # minimum word count to 2 in order to give higher frequency words more weighting
iter = 25  # iterating over the training corpus x times
window = 8

# Initialize Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)
texts = []  # List of lists of tokens


def get_instacities1m():
    # -- LOAD DATA FROM INSTAGRAM --
    posts_text = []
    for city in cities:
        print("Loading InstaCities1M data from " + city)
        for i, file_name in tqdm(enumerate(
                glob.glob(instagram_text_data_path + city + "/*.txt")
        )):
            caption = ""
            filtered_caption = ""
            file = open(file_name, "r")
            for line in file:
                caption = caption + line
            # Replace hashtags with spaces
            caption = caption.replace('#', ' ')
            # Keep only letters and numbers
            for char in caption:
                if char in whitelist:
                    filtered_caption += char

            posts_text.append(filtered_caption.lower())
    return posts_text


def get_mirflickr():
    # -- LOAD DATA FROM MIRFlickr --
    print("Loading MIRFlickr data")
    posts_text = []
    train_half_indices_ints = []
    # Read topics for only retrieval images
    train_half = '../../../datasets/MIRFLICKR25K/train_half.txt'
    with open(train_half) as f:
        train_half_indices = f.readlines()
    for q in train_half_indices:
        train_half_indices_ints.append(int(q))
    for file_name in glob.glob("/home/raulgomez/datasets/MIRFLICKR25K/filtered_topics/*.txt"):
        if int(file_name.split('/')[-1][:-4]) not in train_half_indices_ints:
            continue
        file = open(file_name, "r")
        lines = []
        for line in file:
            line = line.replace('\n', '').replace('\t', '').replace('\r', '')
            line = line.replace('plant_life', 'plant')
            lines.append(line)
        posts_text.append(lines[0].replace(',', ' ') + ' ' + lines[1].replace(',', ' '))
        file.close()
    return posts_text


def get_webvision():
    # -- LOAD DATA FROM WEBVISION --
    posts_text = []
    former_filename = ' '
    print("Loading WebVision data")
    file = open(webvision_text_data_path + 'info/train_meta_list_all.txt', "r")

    for line in file:

        filename = line.split(' ')[0]
        filename = filename.replace('google', 'google_json')
        filename = filename.replace('flickr', 'flickr_json')
        idx = int(line.split(' ')[1])

        if filename != former_filename:
            print(filename)
            json_data = open(webvision_text_data_path + filename)
            d = json.load(json_data)
            former_filename = filename

        caption = ''
        filtered_caption = ''

        if d[idx - 1].has_key('description'): caption = caption + d[idx - 1]['description'] + ' '
        if d[idx - 1].has_key('title'): caption = caption + d[idx - 1]['title'] + ' '
        if d[idx - 1].has_key('tags'):
            for tag in d[idx - 1]['tags']:
                caption = caption + tag + ' '

        # Replace hashtags with spaces
        caption = caption.replace('#', ' ')
        # Keep only letters and numbers
        for char in caption:
            if char in whitelist:
                filtered_caption += char

        posts_text.append(filtered_caption.decode('utf-8').lower())

    return posts_text


class InstaCities1MCorpus:
    def __init__(self):
        pass

    def __iter__(self):
        # -- LOAD DATA FROM INSTAGRAM --
        for city in cities:
            print("Loading InstaCities1M data from " + city)
            for i, file_name in tqdm(enumerate(
                    glob.glob(instagram_text_data_path + city + "/*.txt")
            )):
                caption = ""
                filtered_caption = ""
                file = open(file_name, "r")
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
    # posts_text = get_instacities1m()
    # print("Number of posts: " + str(len(posts_text)))
    # print("Creating tokens")
    # c = 0
    #
    # for t in posts_text:
    #     c += 1
    #     if c % 10000 == 0:
    #         print(c)
    #
    #     try:
    #         t = t.lower()
    #         # Gensim simple_preproces instead tokenizer, 每次产生一个list[str]，每个元素是一个token,把太长或太短的单词忽略掉
    #         tokens = gensim.utils.simple_preprocess(t)
    #         # remove stop words from tokens
    #         stopped_tokens = [i for i in tokens if not i in en_stop]
    #         texts.append(stopped_tokens)
    #     except:
    #         continue
    #
    # posts_text = []

    # corpus
    sentences = InstaCities1MCorpus()
    # Train the model
    print("Training ...")
    if finetune:
        print(model.iter)
        model.train(texts, total_examples=model.corpus_count, epochs=25, compute_loss=False)
    else:
        model = gensim.models.Word2Vec(sentences=sentences, vector_size=size, min_count=min_count, workers=cores,
                                       epochs=iter,
                                       window=window)

    model.save(model_path)
    print("DONE")
