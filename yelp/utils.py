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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Majid Mortazavi"
__email__ = "maj.mortazavi@gmail.com"

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def get_means(data):
    return data.groupby('business_id')['stars'].mean()

def get_counts(data):
    return data.groupby('business_id')['stars'].count()

def get_text_lengh(data):
    data['length'] = data['text'].apply(len)
    return data

def bayesian_mean(arr):
    
    prior = 3.25
    confidence = 50
    
    if not prior or not confidence:
        raise TypeError("Bayesian mean must be computed with m and C")
    return ((confidence * prior + arr.sum()) /
            (confidence + arr.count()))

def get_bayesian_estimates(data):
    return data.groupby('business_id')['stars'].agg(bayesian_mean)

def top_businesses(data, n=10):
    grid   = pd.DataFrame({
                'mean':  get_means(data),
                'count': get_counts(data),
                'bayes': get_bayesian_estimates(data)
             })
    return grid.ix[grid['bayes'].argsort()[-n:]]

def plot_mean_frequency(data):
    
    grid   = pd.DataFrame({
                'Mean Rating':  data.groupby('business_id')['stars'].mean(),
                'Number of Reviewers': data.groupby('business_id')['stars'].count()
             })
    grid.plot(x='Number of Reviewers', y='Mean Rating', kind='hexbin',
              xscale='log', cmap='YlGnBu', gridsize=18, mincnt=1,
              title="Star Ratings by Simple Mean")

    plt.tight_layout()
    plt.show()
    
def target_binning(df):

    df['labels']=pd.cut(df['stars'],
    [-np.inf, 3.2 , np.inf],
    labels=['Negative','Positive']).astype('category')

    return df

def target_to_categorical(data):

    labels = data["labels"].values
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    mapping = dict(zip(le.classes_, range(len(le.classes_))))

    labels = to_categorical(np.asarray(labels))

    return labels, mapping

