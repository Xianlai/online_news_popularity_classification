#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:50:25 2017

@author: LAI
"""

import numpy as np
from sklearn.neighbors import KernelDensity
import math

class NaiveBayes():

    def __init__(self, kernel='gaussian', bandwidth=0.75):
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)

    
    def fit(self, X_train, y_train):
        """ Input: training sets(a list of 2 matrices with labels), validating set
        (1 matrix with labels) and a list of indexes of features to be used. Use the  
        KDE to estimate the pdf of each selected feature in training data set and use
        naive Bayes to classify data point in validating set. 
        Output: the prediction error made on validating set.
        """
        self.lhd_0 = []
        self.lhd_1 = []
        
        X_train_0 = X_train[np.where(y_train==0)[0], :]
        X_train_1 = X_train[np.where(y_train==1)[0], :]
        
        self.n_feats = X_train.shape[1]
        self.prob_prior = [len(X_train_0)/len(X_train), 
                           len(X_train_1)/len(X_train)]
        
        for i in range(self.n_feats):
            self.lhd_0.append(self.kde.fit(X_train_0[:, i].reshape(-1, 1)))
            self.lhd_1.append(self.kde.fit(X_train_1[:, i].reshape(-1, 1)))
        
        return self


    def predict(self, X_test):
        """
        """
        laplace = np.full((X_test.shape[0], ), math.log(0.5))
        probs_0 = np.full((X_test.shape[0], ), self.prob_prior[0])
        probs_1 = np.full((X_test.shape[0], ), self.prob_prior[1])
        
        for k in range(self.n_feats):
            prob_0 = self.lhd_0[k].score_samples(X_test[:,k].reshape(-1, 1))
            prob_1 = self.lhd_1[k].score_samples(X_test[:,k].reshape(-1, 1))
            probs_0 = probs_0 + prob_0 + laplace
            probs_1 = probs_1 + prob_1 + laplace
        
        yHat = 1*(probs_0 > probs_1)

        return yHat, probs_0, probs_1
    
    
    def score(self, X_test, y_test, silent=False):
        """
        """
        yHat, _, _ = self.predict(X_test)
        
        hit = sum(i == j for i, j in zip(yHat, y_test))
        
        perf = hit / X_test.shape[0]
        
        if not silent:
            print("accuracy = {0}".format(perf))
        
        return perf
    
    
    def predict_proba(self, X_test):
        """ make probability estimate for all classes
        """
        
        n_test = X_test.shape[0]
        proba = np.zeros((n_test,2))
        
        _, proba_0, proba_1 = self.predict(X_test)
        
        proba_0 = np.exp(proba_0); proba_1 = np.exp(proba_1)
        proba = np.hstack(proba_0/(proba_0+proba_1), proba_1/(proba_0+proba_1))
        
        return proba
        
        

