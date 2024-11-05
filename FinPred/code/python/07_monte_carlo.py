#!/bin/python

"""
Current Step:
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

def monte_carlo(targ, nt=10):
    mean, std = targ.mean(), targ.std()
    guess = np.random.normal(loc=mean, scale=std, size=(len(targ), nt))
    perf = [performance(targ, guess[:, c]) for c in range(nt)]
    best_perf = min(perf)
    best_guess = guess[:, np.argmin(perf)]
    return best_guess, best_perf

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
        print(tg)
        print(f'tr_mse: {tr_mse:.3f}, ts_mse: {ts_mse:.3f}')
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
        # Monte-Carlo on the train data
        bguess, bperf = monte_carlo(tr_targ, nt=10)
        pcorr = scipy.stats.pearsonr(tr_targ, tr_pred).statistic
        gcorr = scipy.stats.pearsonr(tr_targ, bguess).statistic
        print(f'Train: pred_corr: {pcorr}, rand_corr: {gcorr}')
        # Monte-Carlo on the test data
        bguess, bperf = monte_carlo(ts_targ, nt=10)
        pcorr = scipy.stats.pearsonr(ts_targ, ts_pred).statistic
        gcorr = scipy.stats.pearsonr(ts_targ, bguess).statistic
        print(f'Test: pred_corr: {pcorr}, rand_corr: {gcorr}')
        print('-'*20)
        
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
    
