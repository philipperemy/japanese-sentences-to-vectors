from __future__ import print_function

import os

import skipthoughts.training.tools as st_tools
import skipthoughts.training.train as st_train
import skipthoughts.training.vocab as st_vocab

if __name__ == '__main__':
    os.mkdir('tmp')
    X = ['cat is on the floor.'] * 100
    word_dict, wordcount = st_vocab.build_dictionary(X)
    st_vocab.save_dictionary(word_dict, wordcount, 'tmp/dict.dat')
    st_train.trainer(X, saveto='tmp/toy.npz', dictionary='tmp/dict.dat')
    embed_map = st_tools.load_googlenews_vectors()
    model = st_tools.load_model(embed_map)
    vectors = st_tools.encode(model, X)
    print(vectors)
