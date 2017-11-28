
import re
import numpy as np
import pandas as pd
import logging
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer
from sklearn.manifold import TSNE
import generic_io

LOGGER = logging.getLogger('wordembed')

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='info.log',
                    filemode='w')

def main():

    ####################
    # Load Data From File
    ####################

    LOGGER.info('Loading data')

    file_paths = ['veri/tweets/Italy-earthquake-day1-tweets1.jsonl',
                  'veri/tweets/Italy-earthquake-day1-tweets2.jsonl',
                  'veri/tweets/Italy-earthquake-day2-day3-tweets.jsonl',
                  'veri/tweets/NepalQuake-code-mixed-training-tweets.jsonl']

    data_ = []
    for fpath in file_paths:
        tmp_data = generic_io.load_from_file(fpath, file_format='jsonl')
        LOGGER.info(fpath + 'contains' + str(len(tmp_data)) + 'tweets')
        data_ += tmp_data

    ####################
    # Data Preprocessing
    ####################

    LOGGER.info('Preprocessing')

    tknzr = TweetTokenizer()

    tweetlist = []
    for item in data_:
        item['text'] = re.sub(r'https?[^\s]*', 'urlurlurl', item['text'])
        item['text'] = re.sub(r'(?<!\w)@\w{1,15}(?!\w)', 'usrusrusr', item['text'])
        tokenized_tweet = tknzr.tokenize(item['text'])
        tweetlist.append(tokenized_tweet)

    ####################
    # Word2Vec
    ####################

    LOGGER.info('Training word2vec')

    # train model
    model = Word2Vec(tweetlist, min_count=1)

    # save model
    model.save('veri/models/w2v_model_earthquake_' + str(len(tweetlist)) + '.bin')

    # load model
    # new_model = Word2Vec.load('model.bin')

    ####################
    # t-SNE
    ####################

    LOGGER.info('t-SNE step')

    # summarize vocabulary
    words = list(model.wv.vocab)

    vector_list = []
    for w in words:
        vector_list.append(model[w])

    X = np.array(vector_list)

    tsne = TSNE(n_components=2, init='pca', random_state=0)

    Y = tsne.fit_transform(X)

    generic_io.save_to_file(data_=Y, file_path='veri/tsne/tsne_vectors_eartquake_' + \
                            str(len(tweetlist)) + '.pickle')

    # load t-SNE vectors
    # Y = generic_io.load_from_file(file_path='veri/tsne/tsne_vectors_eartquake_' + str(len(tweetlist)) + '.pickle', file_format='pickle')

    tsne_df = pd.DataFrame(Y, columns=['x', 'y'])
    tsne_df['word'] = words

    tsne_df.to_csv('veri/tsne/tsne_eartquake_' + str(len(tweetlist)) + '.csv', index=False)

    tsne_dict = tsne_df.to_dict(orient='list')
    generic_io.save_to_file(data_=tsne_dict, file_path='veri/tsne_for_chart.json')


if __name__=="__main__":
    main()
