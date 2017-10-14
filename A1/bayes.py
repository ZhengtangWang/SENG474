#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Course: SENG474
# Name:   Zhengtang Wang
# ID:     V00802086

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time


class MyBayesClassifier():
    # For graduate and undergraduate students to implement Bernoulli Bayes
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    def train(self, X, y):
        # Your code goes here.
        # Calculate P(y) for each of the classes (y).
        cls, self._Ncls = np.unique(y, return_counts = True)
        self._Nfeat = len(X)
        self._class_prob = cls/len(y)
        # Calculate P(xi = 0|y) and for each y and every feature xi.
        self._Nfeat = np.zeros((len(self._Ncls), self._Nfeat))
        _, Nfeat = X.shape
        fc = np.zeros((len(self._Ncls), Nfeat)) 
        for i, count in enumerate(X):
        	fc[y[i]] += count # Feature counts
        # Implement additive smoothing with α = 1.
        fc += self._smooth
        total = self._Ncls + (2 * self._smooth)
        # Probability of features.
        denominator = total.reshape(len(total), 1)
        self._feat_prob = fc/denominator
        return 

    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        pred = np.zeros(len(X))
        Xprob = np.ones(self._Ncls)
        for i in range(len(X)):
            s = X[i]
            for j in range(self._Ncls):
                Xprob[j] = self._class_prob[j]
                for f, feature in enumerate(s):
                    if feature == 0:
                        Xprob[j] *= (1 - self._feat_prob[j][f])
       	            else:
       	                Xprob[j] *= (self._feat_prob[j][f])
            pred[i] = cls[Xprob.argmax()]
        return pred

class MyMultinomialBayesClassifier():
    # For graduate students only
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    # Train the classifier using features in X and class labels in Y
    def train(self, X, y):
        # Your code goes here.
        return

    # should return an array of predictions, one for each row in X
    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        return np.zeros([X.shape[0],1])
        


""" 
Here is the calling code

"""

categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a count vectorizer")
t0 = time()

vectorizer = CountVectorizer(stop_words='english', binary=False)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)
print('alpha=%i accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0)))