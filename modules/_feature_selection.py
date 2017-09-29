#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 22:22:23 2017

@author: LAI
"""
from sklearn.model_selection import KFold; kf = KFold(n_splits=3)
import copy
import numpy as np


class FeatureRemoval():
    
    def __init__(self, X, Y, clf):
        """
        """
        self.X = X
        self.Y = Y
        self.clf = clf
        
        self.n_feats = [X.shape[1]] # number of features left in each round
        
        self.feats = list(range(self.n_feats[0]))
        
        self.selections = [self.feats] # features left in each round
        self.criteria = [] # criterion in each round
        print("feature_removal object initiated")
               
        
    def select(self):
        """ this function takes in a training set in numpy ndarray format and a 
        validation set in numpy ndarray format. And perform feature selection based 
        on classification error count. Output the indices of selected features.
        """
        
        #### init the criteria of previous round and current round
        self.criterion_prev = 0
        self.criterion_curr = 9999
        
        while self.n_feats[-1] != 1:
        	#### updating
            """
            if this is not the 1st round of selection, update the results from 
            previous round.
            
            modify this if clause to self.criterion_curr >= self.criterion_prev 
            if we want the selection stop after criterion decrease.
            """
            print("removal begins")

            if self.criterion_curr != 9999: 
                
                # self.criterion_prev = self.criterion_curr
                
                del self.feats[idx]
                feats_curr = copy.deepcopy(self.feats)
                
                self.criteria.append(self.criterion_curr)
                self.selections.append(feats_curr)
                self.n_feats.append(len(self.feats))
            
            else:
                print("-----------------")
                feats_temp = copy.deepcopy(self.feats)
                self.criterion_curr = self.calcCriterion(feats_temp)
                self.criteria.append(self.criterion_curr)
                print("init criterion = {0}".format(self.criterion_curr)) 
                
            #### perform next round of selection
            temp_criteria = []
            
            for i in range(self.n_feats[-1]):
                
                feats_temp = copy.deepcopy(self.feats)
                feats_temp.remove(self.feats[i])

                temp_criteria.append(self.calcCriterion(feats_temp))
                
            self.criterion_curr = max(temp_criteria)
            idx = temp_criteria.index(self.criterion_curr)
            
            print("current criterion = {0}, current n_feats:{1}".format(
                    self.criterion_curr, self.n_feats[-1]-1)) 
        
    
    def calcCriterion(self, feats_temp):
        """
        """
        perfs = []
            
        for fit_idx, pred_idx in kf.split(self.X):
            
            model = self.clf()
            model.fit(self.X[fit_idx.reshape(-1, 1), feats_temp], self.Y[fit_idx])
            
            perf = model.score(self.X[pred_idx.reshape(-1, 1), feats_temp], 
                               self.Y[pred_idx], silent=True)
            
            perfs.append(perf)
            
        criterion = self.expectedLoss(perfs)
            
        return criterion
    
    def expectedLoss(self, perfs):
        """
        """
        bias = 1 - np.mean(perfs)
        var = np.var(perfs)
        loss = bias**2 + var
        return loss
        
        


