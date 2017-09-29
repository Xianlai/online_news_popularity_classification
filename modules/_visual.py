#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 08:07:00 2017

@author: LAI
"""
#### 

from matplotlib import pyplot as plt
import matplotlib as mpl

from matplotlib import gridspec
from matplotlib.colors import ListedColormap

import seaborn as sns; sns.set()
import numpy as np
import math

discrete_BlRd = ListedColormap(['#448afc', '#ed6a6a'])

sm_font = {'fontsize' : 13, 
           'fontname' : 'Arial'}

ax_font = {'fontsize' : 15, 
           'fontweight' : 'bold', 
           'fontname' : 'Arial'}

fig_font = {'fontsize' : 17, 
            'fontweight' : 'bold', 
            'fontname' : 'Arial'}

class Visual():
    
    def __init__(self, n_rows, n_cols, figsize=(12,6)):
        
        plt.rcParams['axes.facecolor'] = '#efefef'
        plt.rcParams['figure.facecolor'] = '#efefef'
        
        self.fig, self.axes = self.base_plot(n_rows, n_cols, figsize)
        self.cm_rgb = ['#448afc', '#ed6a6a']


    def setAxParam(self, ax):
            
        # hide axis ticks
        ax.tick_params(axis="both", which="both", bottom="off", top="off",  
                labelbottom="on", left="off", right="off", labelleft="on",
                colors='#a3a3a3')
    
        # remove axis spines
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(color='#a3a3a3', linewidth=2, alpha=0.3)
                
                
    def base_plot(self, n_rows, n_cols, figsize):
        """ set up the fig and axes
        input: numbers of rows, cols and figsize
        output: fig and axes
        """
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        #fig.patch.set_facecolor('#efefef')  
                                
        if type(axes).__name__ == 'ndarray':
            for ax,cnt in zip(axes.ravel(), range(n_rows * n_cols)):
                self.setAxParam(ax)
                
        else:
            self.setAxParam(axes) 
        
        self.n_rows = n_rows
        
        return fig, axes
        
    
    def hist(self, df_X, sr_Y, feats, show=True):
        """ plot histogram of given features of given dataframe
        """
        
        X = df_X[feats].values
        
        n_class = len(sr_Y.unique())
        Y = sr_Y.values
        
        # set bin sizes
        min_b = math.floor(np.min(X))
        max_b = math.ceil(np.max(X))
        bins = np.linspace(min_b, max_b, 25)
    
        # plottling the histograms
        for lab,col in zip(range(n_class), self.cm_rgb[:n_class]):
            self.axes.hist(X[Y==lab], color=col, label='class %s' %lab,
                           bins=bins, alpha=0.5,)
            
        ylims = self.axes.get_ylim()
    
        # plot annotation
        leg = self.axes.legend(loc='upper right', fancybox=True, fontsize=8)
        leg.get_frame().set_alpha(0.5)
        
        self.axes.set_ylim([0, max(ylims) + 4])
        
        self.axes.set_xlabel(feats, **ax_font)
        self.axes.set_ylabel('count', **ax_font)
        
        self.axes.set_title('Feature Histogram', **fig_font)
        
        # self.fig.tight_layout()
        
        if show:
            plt.show()
    
        
    def hists(self, X, Y, feats, bin_num=25, show=True):
        """ plot histogram of given features of given dataframe
        """
        
        n_class = len(np.unique(Y))

        for ax,cnt in zip(self.axes.ravel(), range(len(feats))):  
            
            feat = feats[cnt]
            # set bin sizes
            min_b = math.floor(np.min(X[:,feat]))
            max_b = math.ceil(np.max(X[:,feat]))
            bins = np.linspace(min_b, max_b, bin_num)
        
            # plottling the histograms
            for lab,col in zip(range(n_class), self.cm_rgb[:n_class]):
                idx = np.where(Y==lab)[0]
                ax.hist(X[idx, feat],
                           color=col,
                           label='class %s' %lab,
                           bins=bins,
                           alpha=0.5,)
                
            ylims = ax.get_ylim()
        
            # plot annotation
            leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
            leg.get_frame().set_alpha(0.5)
            
            ax.set_ylim([0, max(ylims) + 4])
            
            ax.set_xlabel(feat, **ax_font)
        
        for i in range(self.axes.shape[0]):
            self.axes[i][0].set_ylabel('count', **ax_font)
        
        self.fig.suptitle('Feature Histogram', **fig_font)
        
        self.fig.tight_layout()
        
        if show:
            plt.show()
      

        
    def heatmap(self, df_X, sr_Y, feats, bins, show=True):
        """ plot heatmap scatter plot of given 2 features.
        
        """
        
        # prepare data
        X = df_X[feats].values
        X = np.histogram2d(X[:, 0], X[:,1], bins=bins)
        X = X[0]
         
        # Plot heatmap
        
        sns.heatmap(X, annot=True)
        
        # plot annotation
        self.axes.set_title('Heatmap of %s and %s' % (feats[0], feats[1]), **fig_font)
        self.axes.set_xlabel(feats[0], **ax_font)
        self.axes.set_ylabel(feats[1], **ax_font)
        
        if show:
            plt.show()

 
        
    def scatter(self, X, Y, Z, xlabel, ylabel, alpha=0.6, show=True):
        """
        """
        self.axes.scatter(X, Y, c=Z, alpha=alpha, cmap=discrete_BlRd)
        
        # plot annotation
        self.axes.set_xlabel(xlabel, **ax_font)
        self.axes.set_ylabel(ylabel, **ax_font)
        self.axes.set_title('scatter of %s and %s' % (xlabel, ylabel), **fig_font)
        
        if show:
            plt.show()
        
        
    def scatters(self, X, Z, plot_feats, in_feats, alpha=0.6, show=True):
        """
        """
        for ax,cnt in zip(self.axes.ravel(), range(len(plot_feats))):
            
            xx = X[:, plot_feats[cnt][0]]
            yy = X[:, plot_feats[cnt][1]]
            ax.scatter(xx, yy, c=Z, alpha=alpha, cmap=discrete_BlRd)
        
            # plot annotation
            xlabel = in_feats[plot_feats[cnt][0]]
            ylabel = in_feats[plot_feats[cnt][1]]
            
            ax.set_xlabel(xlabel, **ax_font)
            ax.set_ylabel(ylabel, **ax_font)
            
        self.fig.suptitle('Scatters of selected features', **fig_font)
        self.fig.tight_layout()
        
        if show:
            plt.show()
        
        
    def line(self, X, Y, xlabel, ylabel, title, show=True):
        """
        """
        
        # plot line and nodes
        self.axes.plot(X, Y, '-', color=self.cm_rgb[0], linewidth=2)
        self.axes.plot(X, Y, 'o', color=self.cm_rgb[0], linewidth=2)
        
        # plot annotation
        self.axes.set_xlabel(xlabel, **ax_font)
        self.axes.set_ylabel(ylabel, **ax_font)
        self.axes.set_title(title, **fig_font)
                    
        if show:
            plt.show()

    def lines(self, x, Y, show=True):
        """
        """
        self.axes.plot(x, Y[0], '-', color='black', label='expected_loss')
        self.axes.plot(x, Y[0], 'o', color='black')
        
        self.axes.plot(x, Y[1], '--', color='#448afc', label='rescaled bias')
        self.axes.plot(x, Y[1], 'o', color='#448afc')

        self.axes.plot(x, Y[2], '--', color='#ed6a6a', label='rescaled variance')
        self.axes.plot(x, Y[2], 'o', color='#ed6a6a')

        #self.axes.vlines((4.5, 9.5, 14.5, 19.5, 24.5), ymin=Y[0].min(), 
                         #ymax=Y[0].max(), linestyles='dashed', linewidth=1)

        self.axes.legend(loc='upper right')
        self.axes.set_xlabel('classifiers/space', **ax_font)
        self.axes.set_ylabel('expected_loss', color='black', **ax_font)
        self.axes.grid(color='black', linestyle='--', linewidth=1)
        self.axes.tick_params('y', colors='black')
        self.axes.set_xticks(range(len(x)))
        self.fig.suptitle('Performance Evaluation', **fig_font)
        self.fig.tight_layout()

        if show:
            plt.show()
        

    def parallel_coordinates(self, df, Z_col, spa, cmap='RdYlGn', show=True):
        """
        """

        # x-axis
        params = list(df)
        params.remove(Z_col)
        X = range(len(params))
        
        df_grey = df.copy()
        
        for col in params:
            lim = df_grey[col].max() - df_grey[col].min()
            df_grey[col] = 5 * (df_grey[col] - df_grey[col].min()) / lim
        
        df_grey = df_grey.round(6)
        df_color = df_grey.iloc[:10].copy()
        
        # z-axis
        Z_min = df_color[Z_col].min()
        Z_max = df_color[Z_col].max()
        Z = df_color[Z_col].apply(lambda x: (x-Z_min)/(Z_max - Z_min))
        
        # y-axis
        Y_grey = df_grey.drop(Z_col, axis=1).values # grey lines
        Y_color = df_color.drop(Z_col, axis=1).values # color lines
        
        shift = np.zeros(Y_color.shape)
        
        for n in range(shift.shape[0]):
            shift[n,:] = shift[n,:] + 0.02 * n
            
        Y_color = Y_color + shift

        # plot vlines
        for i in X:
            
            self.axes.axvline(i, linewidth=1, color='black')
        
        # plot grey lines
        for n in range(Y_grey.shape[0]):
            
            Y = Y_grey[n]
            self.axes.plot(X, Y, '-', color='grey', linewidth=0.5, alpha=0.5)
        
        # plot color lines
        cm = plt.get_cmap(cmap)
        
        for n in range(Y_color.shape[0]):
            Y = Y_color[n]
            z_val = Z.iloc[n]
            
            self.axes.plot(X, Y, '-', color=cm(1-z_val), linewidth=2)
            self.axes.plot(X, Y, 'o', c=cm(1-z_val), ms=7)
    

        # plot annotations
        self.axes.set_xticks(X)
        self.axes.set_xticklabels(params, **ax_font)
        self.axes.grid(b=False)
        #self.axes.set_xticks([])
        self.axes.set_yticks([])
        
        #self.axes.set_xlabel('parameters', **ax_font)
        
        # plot table
        celltext = df.drop(Z_col, axis=1).iloc[:10].values
        rows = df[Z_col].round(6).iloc[:10].values
        
        plt.table(cellText=celltext, cellLoc='center', rowLabels=rows, 
                  rowColours=cm(1-Z.values), colLabels=params,
                  loc='upper left', bbox=[-0.2, 0, 0.18, 1])
        
        """
        # plot color bar
        norm = mpl.colors.Normalize(Z_min, Z_max)
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        plt.colorbar(sm)
        """
        self.axes.set_title('Parameter Combinations in %s space' %spa, **fig_font)
        self.fig.tight_layout()
        
        if show:
            plt.show()
        
        

    def decision_contour(self, X, Y, clfs, xlabel, ylabel, 
                         alpha_sca=0.6, show=True):
        """
        """
        
        cm = plt.cm.coolwarm
        
        # prepare mesh grid
        sz = 0.02  # step size in the mesh

        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, sz),
                             np.arange(y_min, y_max, sz))
        
        param_type = {0:'default', 1:'tuned'}
        
        for ax, i in zip(self.axes, range(2)):

            clfs[i].fit(X, Y.ravel()) # fit the classifier to training data
            score = clfs[i].score(X, Y.ravel()) # test the clf on the same set
    
            # get the classification probability or confidence for each grid point
            if hasattr(clfs[i], "decision_function"):
                Z = clfs[i].decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clfs[i].predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            
            Z = Z.reshape(xx.shape)

            # plot color grid
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.5)
            
            # plot training data points
            ax.scatter(X[:, 0], X[:, 1], c=Y, alpha=alpha_sca, 
                              cmap=ListedColormap(['#448afc', '#ed6a6a']))
            
            ax.set_xlabel(xlabel,**ax_font )                            
            ax.set_title('Decision Boundries with %s parameters' %param_type[i], 
                         **fig_font)
            ax.text(xx.max() - 0.1, yy.min() + 0.1, ('Score = %.2f' % score),
                    horizontalalignment='right', **ax_font)
            
            if i==0:
                ax.set_ylabel(ylabel, **ax_font)

        self.fig.tight_layout()
        
        if show:
            plt.show()
        
        
    def confusion_hist(self, y_test, y_pred_probs, clf_name, spa_name, show=True):
        """ plot the confusion histogram of FP, TP, TN, FN closewise. 
        in: y_test(the labels:n_samples*1), y_pred_probs(the probabilities of all 
        classes for each data point:n_samples*n_classes), the name of classifier 
        and the name of feature space.
        """
        cmap = plt.cm.PiYG
        predictions = {0:'Negative', 1:'Positive'}
        bounds = np.linspace(-0.5,0.5,11)

        for lbl in [0,1]: # 0 = negative, 1 = positive
            pred = y_pred_probs[y_pred_probs[:,lbl] >= 0.5][:,lbl]
            test = y_test[y_pred_probs[:,lbl] >= 0.5].ravel()
            cfd_err = []
            for i in range(len(test)):
                cfd_err.append((-1)**test[i] * (0.5-pred[i]))
            
            y,x = np.histogram(cfd_err, bins=10, range=(-0.5, 0.5))
            c = np.arange(0, 1, 0.1)
            if lbl == 0: y = -y
            self.axes.bar(left=x[:-1]+0.05, height=y, color=cmap(c), 
                          width=0.1, alpha=0.5)
            
            self.axes.axvline(0.0, linewidth=1, color='black', ls='dashed')
            self.axes.axhline(0.0, linewidth=1, color='black', ls='dashed')

            if lbl == 0: y_txt = y.min()
            else: y_txt = y.max()
            self.axes.text(-0.38, y_txt, 'False %s' % predictions[lbl], 
                    horizontalalignment='center', **ax_font)
            self.axes.text(0.38, y_txt, 'True %s' % predictions[lbl], 
                    horizontalalignment='center', **ax_font)

            self.axes.set_xticks(bounds)
            self.axes.set_xticklabels(np.linspace(-1.0, 0, 11))
            self.axes.set_xlabel('confidence error', **ax_font)
            self.axes.set_title("%s in %s space" %(clf_name, spa_name), 
                                **fig_font)

        if show:
            plt.show()

    def confusion_hists(self, y_test, y_pred_probs, clf_names, spa_names, show=True):
        """ plot the confusion histogram of FP, TP, TN, FN closewise. 
        in: y_test(the labels:n_samples*1), y_pred_probs(the probabilities of all 
        classes for each data point:n_samples*n_classes), the name of classifier 
        and the name of feature space.
        """
        cmap = plt.cm.PiYG
        predictions = {0:'Negative', 1:'Positive'}
        bounds = np.linspace(-0.5,0.5,11)

        for cnt, ax in zip(range(len(clf_names)), self.axes.ravel()):
            for lbl in [0,1]: # 0 = negative, 1 = positive
                pred = y_pred_probs[cnt][y_pred_probs[cnt][:,lbl] >= 0.5][:,lbl]
                test = y_test[y_pred_probs[cnt][:,lbl] >= 0.5].ravel()
                cfd_err = []
                for i in range(len(test)):
                    cfd_err.append((-1)**test[i] * (0.5-pred[i]))
            
                y,x = np.histogram(cfd_err, bins=10, range=(-0.5, 0.5))
                c = np.arange(0, 1, 0.1)
                if lbl == 0: y = -y
                ax.bar(left=x[:-1]+0.05, height=y, color=cmap(c), 
                       width=0.1, alpha=0.7)
            
                ax.axvline(0.0, linewidth=1, color='black', ls='dashed')
                ax.axhline(0.0, linewidth=1, color='black', ls='dashed')

                if lbl == 0: y_txt = y.min()
                else: y_txt = y.max()
                ax.text(-0.3, 0.9*y_txt, 'False %s' % predictions[lbl], 
                               horizontalalignment='center', **sm_font)
                ax.text(0.3, 0.9*y_txt, 'True %s' % predictions[lbl], 
                                horizontalalignment='center', **sm_font)

                ax.set_xticks(bounds)
                ax.set_xticklabels(np.linspace(-1.0, 0, 11))
                ax.set_xlabel('confidence error', **sm_font)
                ax.set_title("%s in %s space" %(clf_names[cnt], spa_names[cnt]), 
                             **fig_font)

        self.fig.suptitle('Confusion Historgams', **fig_font)
        if show:
            plt.show()
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

