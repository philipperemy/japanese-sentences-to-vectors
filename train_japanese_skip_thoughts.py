from __future__ import print_function

from gensim.models import Word2Vec as word2vec

import skipthoughts.training.tools as st_tools
import skipthoughts.training.train as st_train
import skipthoughts.training.vocab as st_vocab


def load_w2v_vectors(path_to_w2v):
    return word2vec.load_word2vec_format(path_to_w2v, binary=False)


if __name__ == '__main__':
    X = ['cat is on the floor.'] * 100
    word_dict, wordcount = st_vocab.build_dictionary(X)
    st_vocab.save_dictionary(word_dict, wordcount, 'dict.dat')
    st_train.trainer(X, saveto='toy.npz', dictionary='dict.dat')
    embed_map = load_w2v_vectors('ja-gensim.50d.data.txt')
    model = st_tools.load_model(embed_map)
    vectors = st_tools.encode(model, X)
    print(vectors)
