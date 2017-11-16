#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 08:07:00 2017

@author: LAI
"""
#### 

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns; sns.set()
import numpy as np
import math

disc_BlRd = ListedColormap(['#448afc', '#ed6a6a'])
sm_font   = {
    'fontsize' : 13, 
    'fontname' : 'Arial'
}
ax_font   = {
    'fontsize' : 15, 
    'fontweight' : 'bold', 
    'fontname' : 'Arial'
}
fig_font  = {
    'fontsize' : 17, 
    'fontweight' : 'bold', 
    'fontname' : 'Arial'
}

class Visual():
    
    def __init__(self, n_rows, n_cols, figsize=(12,6)):
        
        plt.rcParams['axes.facecolor']   = '#efefef'
        plt.rcParams['figure.facecolor'] = '#efefef'
        
        self.fig, self.axes = self.base_plot(n_rows, n_cols, figsize)
        self.cm_rgb         = ['#448afc', '#ed6a6a']


    def setAxParam(self, ax):
            
        # hide axis ticks
        ax.tick_params(
            axis="both", which="both", colors='#a3a3a3', 
            bottom="off", top="off", left="off", right="off",
            labelbottom="on", labelleft="on"
        )
    
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
            for ax in axes.ravel(): self.setAxParam(ax)
        else:
            self.setAxParam(axes) 
        
        self.n_rows = n_rows
        
        return fig, axes

    @staticmethod
    def show():
        plt.show()
        

    def plotHists(self, Xs, Y, dims, spaces, bin_num=25, show=True):
        """ plot histogram of given features of given dataframe
        """
        
        n_class      = len(np.unique(Y))
        n_row, n_col = self.axes.shape

        for i, ax in enumerate(self.axes.ravel()):  
            
            dim   = dims[i % n_col]
            space = spaces[i % n_row]
            # set bin sizes
            min_b = math.floor(np.min(Xs[space][:,dim]))
            max_b = math.ceil(np.max(Xs[space][:,dim]))
            bins  = np.linspace(min_b, max_b, bin_num)
        
            # plottling the histograms
            for lab,col in enumerate(self.cm_rgb[:n_class]):
                idx = np.where(Y==lab)[0]
                ax.hist(
                    Xs[space][idx, dim], color=col, bins=bins, 
                    label='class %s' %lab, alpha=0.5
                )
                
            ylims = ax.get_ylim()
        
            # plot annotation
            leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
            leg.get_frame().set_alpha(0.5)
            
            ax.set_ylim([0, max(ylims) + 4])
            ax.grid(b='off')
        
        for i in range(n_row):
            self.axes[i][0].set_ylabel(spaces[i], **ax_font)
        
        self.fig.suptitle('Feature Hists in Different Spaces', **fig_font)
        self.fig.tight_layout()
        if show:
            plt.show()


    def plotScatters(self, X, Z, plot_feats, in_feats, alpha=0.6, show=True):
        """
        """
        for ax,cnt in zip(self.axes.ravel(), range(len(plot_feats))):
            
            xx = X[:, plot_feats[cnt][0]]
            yy = X[:, plot_feats[cnt][1]]
            ax.scatter(xx, yy, c=Z, alpha=alpha, cmap=disc_BlRd)
        
            # plot annotation
            xlabel = in_feats[plot_feats[cnt][0]]
            ylabel = in_feats[plot_feats[cnt][1]]
            
            ax.set_xlabel(xlabel, **ax_font)
            ax.set_ylabel(ylabel, **ax_font)
            
        self.fig.suptitle('Scatters of selected features', **fig_font)
        self.fig.tight_layout()
        
        if show:
            plt.show()
        

    def plotLines(self, x, Y, show=True):
        """
        """
        self.axes.plot(x, Y[0], '-', color='black', label='expected_loss')
        self.axes.plot(x, Y[0], 'o', color='black', label=None)
        
        self.axes.plot(x, Y[1], '--', color='#448afc', label='mean_accuracy')
        self.axes.plot(x, Y[1], 'o', color='#448afc', label=None)

        self.axes.plot(x, Y[2], '--', color='#ed6a6a', label='variance')
        self.axes.plot(x, Y[2], 'o', color='#ed6a6a', label=None)

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
        

    def plotParallelCoordinates(self, df, Z_col, cmap='RdYlGn', show=True):
        """ plot the paralell coordinates with given DataFrame and the column
        name for colors. 
        """
        # x-axis
        params = list(df)
        params.remove(Z_col)
        X = range(len(params))
        
        df_grey = df.copy()
        
        for col in params:
            lim          = df_grey[col].max() - df_grey[col].min()
            df_grey[col] = 5 * (df_grey[col] - df_grey[col].min()) / lim
        
        df_grey  = df_grey.round(6)
        df_color = df_grey.iloc[:10].copy()
        
        # z-axis
        Z_min = df_color[Z_col].min()
        Z_max = df_color[Z_col].max()
        Z     = df_color[Z_col].apply(lambda x: (x-Z_min)/(Z_max - Z_min))
        
        # y-axis
        Y_grey  = df_grey.drop(Z_col, axis=1).values # grey lines
        Y_color = df_color.drop(Z_col, axis=1).values # color lines
        
        shift   = np.zeros(Y_color.shape)
        
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
            Y     = Y_color[n]
            z_val = Z.iloc[n]
            self.axes.plot(X, Y, '-', color=cm(1-z_val), linewidth=2)
            self.axes.plot(X, Y, 'o', c=cm(1-z_val), ms=7)
    
        # plot annotations
        self.axes.set_xticks(X)
        self.axes.set_xticklabels(params, **ax_font)
        self.axes.grid(b=False)
        self.axes.set_yticks([])
        
        # plot table
        celltext = df.drop(Z_col, axis=1).iloc[:10].values
        rows     = df[Z_col].round(6).iloc[:10].values
        
        plt.table(
            cellText=celltext, cellLoc='center', rowLabels=rows, 
            rowColours=cm(1-Z.values), colLabels=params,
            loc='upper left', bbox=[-0.2, 0, 0.18, 1]
        )

        self.axes.set_title('Parameter Combinations', **fig_font)
        self.fig.tight_layout()
        
        if show:
            plt.show()
        

    def plotCompareDecisionContour(self, X, y, clfs, alpha_sca=0.6, show=True):
        """ plot the decision countours of given classifier of 2 parameter com-
        binations.
        """
        cm         = plt.cm.coolwarm
        param_type = {0:'default', 1:'tuned'}

        # prepare mesh grid
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy       = np.meshgrid(
            np.arange(x_min, x_max, 0.02),
            np.arange(y_min, y_max, 0.02)
        )
        
        for ax, i in zip(self.axes, range(2)):
            clfs[i].fit(X, y.ravel()) # fit the classifier to training data
    
            # get the classification probability or confidence for each grid point
            if hasattr(clfs[i], "decision_function"):
                Z = clfs[i].decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clfs[i].predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            
            Z = Z.reshape(xx.shape)

            # plot color grid
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.5)
            # plot training data points
            ax.scatter(X[:, 0], X[:, 1], c=y, alpha=alpha_sca, 
                       cmap=ListedColormap(['#448afc', '#ed6a6a']))
                                       
            ax.set_title('%s model' %param_type[i], **fig_font)

        self.fig.tight_layout()
        if show:
            plt.show()


    def plotDecisionContours(self, Xs, y, spaces, models, dimPairs, 
                         alpha_sca, show=True):
        """ plot the decision countours of given classifier of 2 parameter 
        combinations.
        """
        cm           = plt.cm.coolwarm
        n_row, n_col = self.axes.shape
        y            = y[:2000]

        for i, ax in enumerate(self.axes.ravel()):
            # prepare mesh grid
            dimPair = dimPairs[i % n_col]
            space   = spaces[i % n_row]
            model   = models[i % n_row]

            X            = Xs[space][:2000, dimPair]
            x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
            xx, yy       = np.meshgrid(
                np.arange(x_min, x_max, 0.02),
                np.arange(y_min, y_max, 0.02)
            )
        
            model.fit(X, y.ravel()) # fit the classifier to training data
    
            # get the classification probability or confidence for each grid point
            if hasattr(model, "decision_function"):
                Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            
            Z = Z.reshape(xx.shape)

            # plot color grid
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.5)
            # plot training data points
            ax.scatter(
                X[:, 0], X[:, 1], c=y, alpha=alpha_sca, 
                cmap=ListedColormap(['#448afc', '#ed6a6a'])
            )
                                
        for i in range(n_row):
            self.axes[i][0].set_ylabel(spaces[i], **ax_font)

        self.fig.suptitle('Decision Contours in Different Spaces', **fig_font)       
        self.fig.tight_layout()

        if show:
            plt.show()
        

    def plotConfusionHists(self, confusionHists, clfNames, spaceNames,
        show=True):
        """ plot the confusion histograms of FP, TP, TN, FN closewise of given 
        fitted models.

        Inputs:
        -------
        fittedModels: the fitted models containing confusion histograms to plot
        """
        cmap        = plt.cm.PiYG
        predictions = {0:'Negative', 1:'Positive'}
        bounds      = np.linspace(-0.5,0.5,11)

        for cnt, ax in enumerate(self.axes.ravel()):
            for lbl in [0,1]:
                y, x = confusionHists[cnt][lbl]
                c    = np.arange(0, 1, 0.1)
                if lbl == 0: y = -y
                ax.bar(
                    left=x[:-1]+0.05, height=y, color=cmap(c), 
                    width=0.1, alpha=0.7
                )
                ax.axvline(0.0, linewidth=1, color='black', ls='dashed')
                ax.axhline(0.0, linewidth=1, color='black', ls='dashed')

                if lbl == 0: y_txt = y.min()
                else: y_txt = y.max()

                ax.text(
                    -0.3, 0.9*y_txt, 'False %s' % predictions[lbl], 
                    horizontalalignment='center', **sm_font
                )
                ax.text(
                    0.3, 0.9*y_txt, 'True %s' % predictions[lbl], 
                    horizontalalignment='center', **sm_font
                )

                ax.set_xticks(bounds)
                ax.set_xticklabels(bounds)
                ax.set_xlabel('confidence error', **sm_font)
                ax.set_title("%s in %s space" %(clfNames[cnt], 
                    spaceNames[cnt]), **ax_font)

        self.fig.tight_layout()

        if show:
            plt.show()
            

