# coding=utf-8
# Copyright 2018 Majid Mortazavi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Majid Mortazavi"
__email__ = "maj.mortazavi@gmail.com"

import string
import os.path
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

def text_clean(text):
    
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    nopunc = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", nopunc)
    list_of_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    doc = ' '.join(w for w in list_of_words)

    return doc
    
def tokenize(df, max_num_words=1000):

    texts = df["clean_text"].values
    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(texts)
    words = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    return words, word_index

def create_embedding_matrix(word_index, embedding_dir, embedding_vec, embedding_dim):

    print(50*'-')
    print('Creating Embedding Matrix...')
    embedding_index = {}

    f = open(os.path.join(embedding_dir, embedding_vec))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embedding_index[word] = coefs
    f.close()

    print('Found Word Vecs: ',len(embedding_index))

    embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))

    for word,i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def get_wordcloud(df, col_to_plot='clean_text_length', target='All', folder_to_write='../results',
                  figure_name='wordcloud.png', show=True):
    
    """Given a panda dataframe, produce wordcloud according to word frequency and sentence label."""

    if target == "Negative":
        background_color = 'black'
    elif target == "Positive":
        background_color = 'white'
    else:
        raise Exception("Please specify a correct target."
                        "Possible values are: 1, 3, 5, 'All'.")

    if target != 'All':
        df = df[df['labels'] == target]

    words = []
    for idx in range(len(df)):
        words.extend(df[col_to_plot].iloc[idx])

    all_words = ' '.join([word for word in words])
    wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=110,
                          background_color=background_color).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.title("{} Star Review".format(target), fontsize=20)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')

    path_to_fig = os.path.abspath(os.path.normpath(os.path.join(folder_to_write, str(target) + figure_name)))
    plt.savefig(path_to_fig, dpi=300, bbox_inches='tight')

    if show:
        plt.show();
    plt.clf()