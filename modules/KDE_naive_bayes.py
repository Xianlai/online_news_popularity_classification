#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:50:25 2017

@author: LAI
"""

import numpy as np
from scipy import stats
import math

class KDENB():

    def __init__(self):
        self.likelihoods_0 = []
        self.likelihoods_1 = []

    
    def fit(self, X, Y):
        """ Input: training sets(a list of 2 matrices with labels), validating set
        (1 matrix with labels) and a list of indexes of features to be used. Use the  
        KDE to estimate the pdf of each selected feature in training data set and use
        naive Bayes to classify data point in validating set. 
        Output: the prediction error made on validating set.
        """
        self.likelihoods_0 = []
        self.likelihoods_1 = []
        
        X_0 = X[[i for i in range(len(Y)) if Y[i] == 0], :]
        X_1 = X[[i for i in range(len(Y)) if Y[i] == 1], :]
        
        self.n_feats = X.shape[1]
        self.prob_prior = [len(X_0)/len(X), len(X_1)/len(X)]
        
        for i in range(self.n_feats):
            self.likelihoods_0.append(stats.gaussian_kde(X_0[:, i]))
            self.likelihoods_1.append(stats.gaussian_kde(X_1[:, i]))

        return self


    def predict(self, X):
        """
        """
        
        YHat = []
        
        for i in range(X.shape[0]):
            
            posterior = [math.log(x) for x in self.prob_prior]
            
            for k in range(self.n_feats):
                posterior[0] += math.log((self.likelihoods_0[k].evaluate(X[i, k])+1.0) / 2)
                posterior[1] += math.log((self.likelihoods_1[k].evaluate(X[i, k])+1.0) / 2)
        
            YHat.append(posterior[0] < posterior[1])
        
        return YHat
    
    
    def score(self, X, Y, silent=False):
        """
        """
        
        YHat = self.predict(X)
        
        hit = sum(i == j for i, j in zip(YHat, Y))
        
        perf = hit / X.shape[0]
        
        if not silent:
            print("accuracy = {0}".format(perf))
        
        return perf
    
    
    def predict_proba(self, X):
        """ make probability estimate for all classes
        """
        
        n_test = X.shape[0]
        
        proba = np.zeros((n_test,2))
        
        for i in range(n_test):
            
            posterior = [math.log(x) for x in self.prob_prior]
            
            for k in range(self.n_feats):
                posterior[0] += math.log((self.likelihoods_0[k].evaluate(X[i, k])+1.0) / 2)
                posterior[1] += math.log((self.likelihoods_1[k].evaluate(X[i, k])+1.0) / 2)
        
            #proba[i, :] = posterior[0], posterior[1]
            prob_0, prob_1 = math.exp(posterior[0]), math.exp(posterior[1])
            proba[i, :] = prob_0 / (prob_0 + prob_1), prob_1 / (prob_0 + prob_1)
        
        return proba
        
        