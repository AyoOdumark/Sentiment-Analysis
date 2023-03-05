from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import string
import re
import torch


def read_table(path, names):
    data = pd.read_table(path, header=None, names=names)
    return data


def get_features_and_labels(arr):
    X = arr[:, 1]
    y = arr[:, 0]
    return X, y


def remove_punctuations(arr):
    new_arr = []
    pattern = r"[{}]".format(string.punctuation)
    for sentence in arr:
        new_arr.append(re.sub(pattern, "", sentence.lower()))
    return new_arr


def tokenize(arr):
    arr_length = len(arr)
    for i in range(arr_length):
        arr[i] = word_tokenize(arr[i])
    return arr


def remove_stopwords(arr):
    stop_words = set(stopwords.words("english"))
    filtered_arr = []
    for sentence in arr:
        filtered_arr.append([word for word in sentence if word not in stop_words])
    return filtered_arr


def create_vocab_with_id(arr):
    word_to_ix = {}
    for sentence in arr:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def create_dtm(arr, vocabulary):
    dtm = np.zeros((len(arr), len(vocabulary)), dtype=np.float32)
    for i in range(len(arr)):
        for word in arr[i]:
            if word in vocabulary:
                dtm[i, vocabulary[word]] += 1
    return dtm


def word2vec(word, vocabulary):
    vec = torch.zeros((1, len(vocabulary)))
    word_idx = vocabulary[word]
    vec[0][word_idx] = 1
    return vec





