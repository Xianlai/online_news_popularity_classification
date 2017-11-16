#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 08:07:00 2017

@author: LAI
"""
####
import numpy as np
import pandas as pd
import math
import sklearn.preprocessing as prep
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from Visual import Visual as V

class LearningPipe():

    """
    This is a wrapper class based on several sklearn models to perform feature
    selection, grid search, cross validation and evaluation jobs. And provides
    methods to visualize parameter tuning and decision boundaries.
    """

    def __init__(self, clf, clf_name, Xs, y):
        self.clf      = clf
        self.clf_name = clf_name
        self.Xs       = Xs
        self.y        = y
        self.kf       = KFold(n_splits=5)
        self.spaces   = [
            'original', 'PCA', 'sparsePCA', 'factorAnalysis', 'NMF'
        ]

    @staticmethod
    def featureSelection(X_train, X_test, y, modelType='tree'):
        """ a static method using either linear svm or extra tree to select 
        relevent features for later model fitting.

        Inputs:
        -------
        X_train: The training feature matrix in one space
        X_test: The testing feature matrix in one space
        y: the label matrix
        modelType: either 'tree' or 'lsvc', default as 'tree'

        Outputs:
        -------
        The new feature matrix with selected features.
        """
        if modelType == 'lsvc':
            clf = LinearSVC(C=0.01, penalty="l1", dual=False)
        if modelType == 'tree':
            clf = ExtraTreesClassifier()

        model = SelectFromModel(clf.fit(X_train, y), prefit=True)

        return model.transform(X_train), model.transform(X_test)


    @staticmethod
    def polynomializeFeatures(X_train, X_test, n_degree=2):
        """ a static method transforming the features into polynomal space 
        in order to catpure non-linear relationships in later learning process.

        Inputs:
        -------
        X_train: The training feature matrix in one space
        X_test: The testing feature matrix in one space
        n_degree: The degree of polynomial spaces

        Outputs:
        -------
        The transformed training and testing feature matrix.
        """
        poly         = prep.PolynomialFeatures(n_degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly  = poly.fit_transform(X_test)
        return X_train_poly, X_test_poly


    def pseudoProba(self, dists):
        """ Convert the distance from test data point to decision boundary to 
        pseudo probability using sigmoid function.

        Inputs:
        -------
        dists: The distances from each test data point to decision boundary 
        in discriminative classifiers.

        Outputs:
        -------
        The converted pseudo probability for each data point.
        """
        sigmoid     = lambda x: 1 / (1 + math.exp(-x))
        n_samples   = len(dists)
        pseudoProba = np.empty((n_samples,2))

        for i in range(n_samples):
            pseudoProba[i] = [sigmoid(dists[i]), 1 - sigmoid(dists[i])]

        return pseudoProba


    def _getConfusionHist(self, fittedModel, X_test, y_test):
        """ Calculate confusion histogram(histogram version of confusion matrix) 
        using predict probability of generative model or distance to decision
        boundary of discriminative model.

        Inputs:
        -------
        fittedModel: The classifier already fitted on training dataset
        X_test: the feature matrix for testing
        y_test: the label matrix for testing

        Outputs:
        -------
        The confusion histogram of both positive and negative predictions.
        """
        # fetch the prediction probabilities or pseudo probabilities of each
        # data point from model.
        if hasattr(fittedModel, "predict_proba"):
            yHatProbs = fittedModel.predict_proba(X_test)
        else:
            yHatDists = fittedModel.decision_function(X_test)
            yHatProbs = self.pseudoProba(yHatDists)

        # calculate the confidence error of each data point. For example, if
        # a prediction is <negative=0.7, positive=0.3> while the true label is
        # positive, then the confidence error is -0.2 because probability for 
        # positive has to reach at least 0.5 to make correct prediction. Thus 
        # this prediction is a case of false negative.
        confusionHist = []

        for lbl in [0,1]: # 0 = negative, 1 = positive
            mask          = yHatProbs[:, lbl] >= 0.5
            preds, labels = yHatProbs[mask], y_test[mask]
            cfd_err       = []

            for i in range(len(preds)):
                cfd_err.append((-1)**labels[i] * (0.5 - preds[i]))

            binDensity,binEdge = np.histogram(
                cfd_err, bins=10, range=(-0.5, 0.5), density=True
            )

            confusionHist.append([binDensity,binEdge])

        return np.array(confusionHist)


    def evaluate(self, param, X, y):
        """ Evaluate the accuracies of the current classifier with given
        parameter combination using 5-fold cross validation. Then calcu-
        late the average accuracy, variance and expected loss based on 
        the results of cross validation.

        Inputs:
        -------
        param: parameter combination for classifier
        X: the features to fit
        y: the labels to fit

        Outputs:
        -------
        performance: the average accuracy, variance and expected loss
        """
        confusionHists = []
        scores         = []

        for train_idx, test_idx in self.kf.split(X):
            model = self.clf(**param)
            model.fit(X[train_idx], y[train_idx].ravel())
            scores.append(
                model.score(X[test_idx], 
                y[test_idx].ravel())
            )
            confusionHists.append(
                self._getConfusionHist(model, 
                X[test_idx], 
                y[test_idx].ravel())
            )

        mean        = np.mean(scores)
        var         = np.var(scores)
        loss        = (1 - mean)**2 + var
        performance = (mean, var, loss)

        return performance, np.mean(confusionHists, axis=0)


    def gridSearching(self, params={}):
        """ With the given values of parameters, this function will gene-
        rate a parameter grid using sklearn and evaluate each of the para-
        meter combination in each feature spaces and record the performances
        in self.results as a DataFrame.

        Inputs:
        -------
        params: possible values for each parameter
        """
        paramsGrid = ParameterGrid(params)
        results    = []
        print("number of combinations in each space: %d" %len(paramsGrid))
        
        for space in self.spaces:
            print("searching space: %s" %space)
            for cnt, param in enumerate(paramsGrid):
                print(cnt)
                performance, confusionHist = self.evaluate(
                    param, self.Xs[space], self.y
                )
                mean, var, loss = performance
                param.update({
                    'space':space, 'mean':mean,
                    'variance':var, 'expectedLoss':loss,
                    'confusionHist':confusionHist
                })
                results.append(param)

        df             = pd.DataFrame(results).sort_values('expectedLoss')
        self.results   = df.reset_index(drop=True)
        self.bestPerfs = self._getBestPerfs()


    def _getBestPerfs(self):
        """ 
        output:
        -------
        A dataframe contains the best results in each feature space
        """
        bestPerfs    = []
        resultGroups = self.results.groupby('space')
        cols         = [
            'space', 'mean', 'variance', 'expectedLoss', 'confusionHist'
        ]

        for space in self.spaces:
            bestPerf = resultGroups\
                .get_group(space)[cols]\
                .to_dict('records')[0]
            bestPerfs.append(bestPerf)

        bestPerfs        = pd.DataFrame(bestPerfs)
        bestPerfs['clf'] = self.clf_name

        return bestPerfs


    def plotParamGrid(self, params, replace=None):
        """ use parallel coordinates plotting to visualize the performance
        (expected loss) of each parameter combination in each state.  

        Inputs:
        -------
        params: possible values for each parameter
        """
        results = self.results.copy()\
            .drop(['mean', 'variance', 'confusionHist'], axis=1)

        if replace:
            results = results.rename(columns=replace)
            for k, v in replace.items(): params[v] = params.pop(k)

        # because of the different scale of values in different parameter,
        # we will replace the value with its index in the given value range
        for col in results.columns:
            if col in params.keys():
                results[col] = results[col].apply(\
                    lambda x: params[col].index(x)
                )

        results['space'] = results['space'].apply(\
            lambda x: self.spaces.index(x)
        )

        v = V(1, 1, figsize=(12, 6))
        v.plotParallelCoordinates(results, 'expectedLoss')


    def compareDecisionContour(self, space, dimPair, alpha=0.6):
        """ plot and compare the decision contours of current classifier 
        before and after tuning.

        Inputs:
        -------
        space: The feature space to visualize
        dimPair: The 2 dimensions in feature space to visualize
        alpha: the alpha of scatter plot of training data points
        """
        cols = [
            'space', 'mean', 'variance', 'expectedLoss', 'confusionHist'
        ]

        tunedParam = self.results.groupby('space')\
            .get_group(space)\
            .drop(cols, axis=1)\
            .to_dict('records')[0]

        clfs = [self.clf(), self.clf(**tunedParam)]
        X    = self.Xs[space][:5000, dimPair]
        y    = self.y[:5000]

        v = V(1, 2, figsize=(12, 6))
        v.plotCompareDecisionContour(X, y, clfs, alpha_sca=alpha, show=True)


    def plotDecisionContours(self, dimPairs, n_col, alpha=0.1, show=True):
        """ plot the decision contours in given pairs of dimensions in all
        feature spaces

        Inputs:
        -------
        dimPairs: The pairs dimensions to visualize
        n_col: how many columns of axes in the figure
        alpha: the alpha of scatter plot of training data points
        """
        models = []
        cols   = [
            'space', 'mean', 'variance', 'expectedLoss', 'confusionHist'
        ]

        for space in self.spaces:
            tunedParam = self.results.groupby('space')\
                .get_group(space)\
                .drop(cols, axis=1)\
                .to_dict('records')[0]

            models.append(self.clf(**tunedParam))
        
        v = V(5, n_col, figsize=(15, 16))
        v.plotDecisionContours(
            self.Xs, self.y, 
            self.spaces, models, dimPairs,
            alpha, show
        )


