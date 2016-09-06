from __future__ import print_function

import re

from gensim.models import Word2Vec as word2vec

import skipthoughts.training.tools as st_tools
import skipthoughts.training.train as st_train
import skipthoughts.training.vocab as st_vocab

#SENTENCES_FILENAME = 'jawiki-latest-text-sentences-tokens.txt'
W2V_FILENAME = 'ja-gensim.50d.data.txt'


SENTENCES_FILENAME = 'test.txt'


def load_w2v_vectors(path_to_w2v):
    return word2vec.load_word2vec_format(path_to_w2v, binary=False)


if __name__ == '__main__':
    with open(SENTENCES_FILENAME, 'r') as inp:
        X = []
        print('** Reading the sentences from <{}> ...'.format(SENTENCES_FILENAME), end="")
        for line in inp.readlines():
            if not re.match('[a-zA-Z]+', line):
                X.append(line.strip())
        print(' [DONE]')
        print('** Loaded {} sentences.'.format(len(X)))

        print('** Building vocabulary ...', end="")
        word_dict, wordcount = st_vocab.build_dictionary(X)
        print(' [DONE]')

        print('** Saving vocabulary ...', end="")
        st_vocab.save_dictionary(word_dict, wordcount, 'dict.dat')
        print(' [DONE]')

        print('** Starting training ...', end="")
        st_train.trainer(X, saveto='toy.npz', dictionary='dict.dat')
        print(' [DONE]')

        print('** Loading w2v vectors from <{}> ...'.format(W2V_FILENAME), end="")
        embed_map = load_w2v_vectors(W2V_FILENAME)
        print(' [DONE]')

        print('** Loading model ...', end="")
        model = st_tools.load_model(embed_map)
        print(' [DONE]')

        print('** Starting to encode the sentences ...', end="")
        vectors = st_tools.encode(model, X)
        print(' [DONE]')
        print(vectors)
