#!/bin/python

"""
Current Step:
Plot the following:
    Top 10 important features
    Mean of group importances
    Max of group importances

Last Step:
Sorting the feature imporances descending:
    Top 10 important features
    Mean of group importances
    Max of group importances

Last Step:
Monte-Carlo simulation between random guess with a Gaussian Distribution vs
proposed predictive model

Last Step:
Plot the train and test results in different directories

Last Step:
o Find targets and their corresponding features
o Classify by pipeline of StandardScaler, PCA, KernelRidge
o Print the target name, number of available samples,
  number of available features, cross validation for regression metrics


Last Step:
o Find the highly missing features (more than some threshold)
o Find the features' types: categorical or continues
o Find the highly missing targets (more than some threshold)
o Find the targests' types: categorical or continues
o Find the balance of categorical targets

Last Step:
o Modify columns by moving identifiers and features to the beginning and
  targets to the end.
o (IS NOT DONE) Removing features with more than 90% missing values.
o (IS NOT DONE) Zero-filling the missing data.
o Saving the new Excel file and Pickle files.
"""

# Imports
#
import sys
import os
import time
import pickle
import numpy as np
# import matplotlib.pyplot as plt
import pandas
import scipy
import scipy.stats
import sklearn
import sklearn.decomposition, sklearn.preprocessing, sklearn.impute
import sklearn.pipeline, sklearn.kernel_ridge, sklearn.model_selection
import sklearn.ensemble, sklearn.inspection
import multiprocessing as mlps
import plotly
import plotly.express as px
import plotly.subplots
import plotly.graph_objects as go

# Globals
#
br = breakpoint
e = lambda: os._exit(0)
NFEAT = 10
TRAIN_TEST_IFILE = '../../data/train_test.pckl'
FEAT_IMPORT_IFILE = '../../output/features_importances_mva_t3_1.pckl'
PLOT_ODIR = '../../output/plots/feature_importance/'

def main(argv):
    with open(TRAIN_TEST_IFILE, mode='rb') as f:
        trts = pickle.load(f)
    with open(FEAT_IMPORT_IFILE, mode='rb') as f:
        fimp = pickle.load(f)
    plot_top10_odir = os.path.join(PLOT_ODIR, 'top10iv')
    plot_group_mean_odir = os.path.join(PLOT_ODIR, 'group_mean')
    plot_group_max_odir = os.path.join(PLOT_ODIR, 'group_max')
    os.makedirs(plot_top10_odir, exist_ok=True)
    os.makedirs(plot_group_mean_odir, exist_ok=True)
    os.makedirs(plot_group_max_odir, exist_ok=True)
    for tg in fimp:
        if tg != 'mva_t3_1':
            continue
        print(tg)
        feat_names = trts[tg]['feat_names']
        # construct data
        pi = fimp[tg]['feat_importance']
        importances_mean_indices = pi.importances_mean.argsort()[::-1]
        top_feats = {feat_names[c]: pi.importances_mean[c]
                     for c in importances_mean_indices}
        # print(f'top 10 important features (descending):\n'
        #       f'{[list(top_feats.keys())[k] for k in range(10)]}')
        # plot top 10 IVs
        title = f'Target: {tg}, Top 10 important IVs'
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        x = list(top_feats.keys())[:10]
        y = list(top_feats.values())[:10]
        fig.add_trace(go.Bar(x=x, y=y, name='top10iv',
                             text=np.round(y, 4)),
                      row=1, col=1)
        fig.update_layout(title=title, font={'size': 16})
        fig.update_layout(xaxis_title='Independent Variable',
                          yaxis_title='Importance wrt. Max MSE')
        plot_fname = os.path.join(plot_top10_odir, f'{tg}.html')
        fig.write_html(plot_fname)
        # plot mean groups
        title = f'Target: {tg}, Groups Mean'
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        x = list(fimp[tg]['mean_score'].keys())
        y = list(fimp[tg]['mean_score'].values())
        fig.add_trace(go.Bar(x=x, y=y, name='group_mean',
                             text=np.round(y, 4)),
                      row=1, col=1)
        fig.update_layout(title=title, font={'size': 16})
        fig.update_layout(xaxis_title='Group',
                          yaxis_title='Mean of Features Importance')
        plot_fname = os.path.join(plot_group_mean_odir, f'{tg}.html')
        fig.write_html(plot_fname)
        # plot mean groups
        title = f'Target: {tg}, Groups Max'
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        x = list(fimp[tg]['max_score'].keys())
        y = list(fimp[tg]['max_score'].values())
        fig.add_trace(go.Bar(x=x, y=y, name='group_max',
                             text=np.round(y, 4)),
                      row=1, col=1)
        fig.update_layout(title=title, font={'size': 16})
        fig.update_layout(xaxis_title='Group',
                          yaxis_title='Max of Features Importance')
        plot_fname = os.path.join(plot_group_max_odir, f'{tg}.html')
        fig.write_html(plot_fname)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
    
