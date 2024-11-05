#!/bin/python

"""
Current Step:
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
import sklearn
import sklearn.decomposition, sklearn.preprocessing, sklearn.impute
import sklearn.pipeline, sklearn.kernel_ridge, sklearn.model_selection
import sklearn.ensemble
import multiprocessing as mlps
import plotly
import plotly.express as px
import plotly.subplots
import plotly.graph_objects as go

# Globals
#
br = breakpoint
FEATS_IFILE = '../../data/feats.pckl'
TRAIN_TEST_IFILE = '../../data/train_test.pckl'
RESULT_IFILE = '../../output/cross_validate_result_scale_no_pca_no.pckl'
PLOT_ODIR = '../../output/plots/'

def performance(targ, pred):
    # r2 = sklearn.metrics.r2_score(targ, pred)
    mse = sklearn.metrics.mean_squared_error(targ, pred)
    return mse

def clean_feats_targets(feat, targ):
    # rescaling the targets
    # targ = sklearn.preprocessing.StandardScaler(
    #     with_mean=True, with_std=True).fit_transform(
    #         targ.reshape(-1, 1)).flatten()
    # Fill missing values with the mean of features
    impute = sklearn.impute.SimpleImputer(missing_values=np.nan,
                                          strategy='mean')
    feat = impute.fit_transform(feat)
    return feat, targ

def main(argv):
    with open(FEATS_IFILE, mode='rb') as f:
        data = pickle.load(f)
    with open(TRAIN_TEST_IFILE, mode='rb') as f:
        trts = pickle.load(f)
    with open(RESULT_IFILE, mode='rb') as f:
        results = pickle.load(f)
    # output plots directories
    tr_plots_odir = os.path.join(PLOT_ODIR, 'train')
    ts_plots_odir = os.path.join(PLOT_ODIR, 'test')
    # os.makedirs(tr_plots_odir, exist_ok=True)
    # os.makedirs(ts_plots_odir, exist_ok=True)
    # Single Processing
    for res in results:
        tg, est, tr_mse, ts_mse = (res['tg'], res['estimator'],
                                   res['tr_mse'], res['ts_mse'])
        print(f'{tg}, tr_mse: {tr_mse:.3f}, ts_mse: {ts_mse:.3f}')
        # make output directories
        tr_plots_tg_odir = os.path.join(tr_plots_odir, tg)
        ts_plots_tg_odir = os.path.join(ts_plots_odir, tg)
        os.makedirs(tr_plots_tg_odir, exist_ok=True)
        os.makedirs(ts_plots_tg_odir, exist_ok=True)
        # construct data
        tg_data = data[data[tg].notna()]
        feat_names = trts[tg]['feat_names']
        tr_companies = trts[tg]['train']
        ts_companies = trts[tg]['test']
        tr_data = tg_data[tg_data['ticker'].isin(tr_companies)]
        ts_data = tg_data[tg_data['ticker'].isin(ts_companies)]
        tr_targ, tr_feat = tr_data[tg].values, tr_data[feat_names].values
        ts_targ, ts_feat = ts_data[tg].values, ts_data[feat_names].values
        tr_feat, tr_targ = clean_feats_targets(tr_feat, tr_targ)
        ts_feat, ts_targ = clean_feats_targets(ts_feat, ts_targ)
        tr_pred, ts_pred = est.predict(tr_feat), est.predict(ts_feat)
        # For train companies
        for comp in tr_companies:
            x = tg_data[tg_data['ticker'] == comp]['year'].values
            indices = np.arange(tr_data.shape[0])[tr_data['ticker']==comp]
            targ, pred = tr_targ[indices], tr_pred[indices]
            # performance
            mse = performance(targ, pred)
            # plot
            title = f'Company: {comp}, Target: {tg}, MSE: {mse:.3f}'
            fig = plotly.subplots.make_subplots(rows=1, cols=1)
            fig.add_trace(go.Scatter(x=x, y=targ, name='actual'),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=pred, name='predicted'),
                          row=1, col=1)
            fig.update_layout(title=title)
            fig.update_layout(xaxis_title='Year', yaxis_title=f'{tg}')
            fig.write_html(os.path.join(tr_plots_tg_odir, f'{comp}.html'))
        # For test companies
        for comp in ts_companies:
            x = tg_data[tg_data['ticker'] == comp]['year'].values
            indices = np.arange(ts_data.shape[0])[ts_data['ticker']==comp]
            targ, pred = ts_targ[indices], ts_pred[indices]
            # performance
            mse = performance(targ, pred)
            # plot
            title = f'Company: {comp}, Target: {tg}, MSE: {mse:.3f}'
            fig = plotly.subplots.make_subplots(rows=1, cols=1)
            fig.add_trace(go.Scatter(x=x, y=targ, name='actual'),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=pred, name='predicted'),
                          row=1, col=1)
            fig.update_layout(title=title)
            fig.update_layout(xaxis_title='Year', yaxis_title=f'{tg}')
            fig.write_html(os.path.join(ts_plots_tg_odir, f'{comp}.html'))
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
    
