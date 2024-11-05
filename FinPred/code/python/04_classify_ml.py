#!/bin/python

"""
Current Step:
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
# TARGETS_IFILE = '../../data/targets.pckl'
FEATS_IFILE = '../../data/feats.pckl'
TRAIN_TEST_IFILE = '../../data/train_test.pckl'
RESULT_OFILE = '../../output/cross_validate_result.pckl'
# RESULT_ODIR = '../../output/targets_results/'
NPCA = 100
NPS = 4

def get_pipeline():
    # clf = sklearn.kernel_ridge.KernelRidge()
    # clf = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
    clf = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    # clf = sklearn.svm.SVR()
    ppl = sklearn.pipeline.Pipeline(steps = [
        ('scaler',
         sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)),
        ('pca', sklearn.decomposition.PCA(n_components=NPCA)),
        ('clf', clf)])
    return ppl

def clean_feats_targets(feat, targ):
    # rescaling the targets
    targ = sklearn.preprocessing.StandardScaler(
        with_mean=True, with_std=True).fit_transform(
            targ.reshape(-1, 1)).flatten()
    # Fill missing values with the mean of features
    impute = sklearn.impute.SimpleImputer(missing_values=np.nan,
                                          strategy='mean')
    feat = impute.fit_transform(feat)
    return feat, targ

def cross_validate(feat, targ):
    """
    data: Pandas' data frame
    feat_cols: a list of feature names
    tg: the current target that should be predicted
    """
    feat, targ = clean_feats_targets(feat, targ)
    # get the pipeline (estimator)
    ppl = get_pipeline()
    result = sklearn.model_selection.cross_validate(
        ppl, feat, targ, cv=5,
        scoring='neg_mean_squared_error',
        # scoring='r2',
        return_train_score=True, return_estimator=True, return_indices=False)
    # print(f'Len: {targ.shape[0]}, feat: {feat.shape[1]}, '
    #       f'Train: {-np.round(result["train_score"].mean(), 2)}, '
    #       f'Test: {-np.round(result["test_score"].mean(), 2)}',
    #       flush=True)
    return result

def performance(targ, pred):
    # r2 = sklearn.metrics.r2_score(targ, pred)
    mse = sklearn.metrics.mean_squared_error(targ, pred)
    return mse

def validate(est, feat, targ):
    feat, targ = clean_feats_targets(feat, targ)
    pred = est.predict(feat)
    scores = performance(targ, pred)
    return scores

def train_validate(tr_feat, tr_targ, ts_feat, ts_targ):
    # training by cross validation
    result = cross_validate(tr_feat, tr_targ)
    # get best model index on the test set
    # best_index = result['test_score'].argmax()
    best_index = result['test_score'].argmin()
    best_estimator = result['estimator'][best_index]
    # get best estimator's score on test data
    tr_score = validate(best_estimator, tr_feat, tr_targ)
    ts_score = validate(best_estimator, ts_feat, ts_targ)
    # Plot some time-series for companies
    # plot_ts(train_data)
    res = {'tr_mse': tr_score, 'ts_mse': ts_score,
           'estimator': best_estimator}
    return res

def process(data, trts, tg):
    t0 = time.time()
    print(tg, end=', ')
    tg_data = data[data[tg].notna()]
    feat_names = trts[tg]['feat_names']
    tr_companies = trts[tg]['train']
    ts_companies = trts[tg]['test']
    tr_data = tg_data[tg_data['ticker'].isin(tr_companies)]
    ts_data = tg_data[tg_data['ticker'].isin(ts_companies)]
    tr_targ, tr_feat = tr_data[tg].values, tr_data[feat_names].values
    ts_targ, ts_feat = ts_data[tg].values, ts_data[feat_names].values
    result = train_validate(tr_feat, tr_targ, ts_feat, ts_targ)
    result.update({'tg': tg})
    print(f'Dur: {time.time() - t0:.0f} s, '
          f'Shape: ({tg_data.shape[0]}, {len(feat_names)}), '
          f'TrMSE: {np.round(result["tr_mse"], 2)}, '
          f'TsMSE: {np.round(result["ts_mse"], 2)}')
    return result

def main(argv):
    with open(FEATS_IFILE, mode='rb') as f:
        data = pickle.load(f)
    # with open(TARGETS_IFILE, mode='rb') as f:
    #     trgs = pickle.load(f)
    # feat_cols = [col for col in data.columns
    #              if not ((col in IDENTIFIERS) | (col in trgs))]
    with open(TRAIN_TEST_IFILE, mode='rb') as f:
        trts = pickle.load(f)
    res = []
    # # Single Processing
    # for tg in list(trts.keys())[:3]:
    #     res.append(process(data, trts, tg))
    # Multiprocessing
    with mlps.pool.Pool(NPS) as pool:
        res = pool.starmap(process,
                           [(data, trts, tg) for tg in list(trts.keys())[:3]])
    # Save the results
    print(f'Saving the results as {RESULT_OFILE} ...')
    with open(RESULT_OFILE, mode='wb') as f:
        pickle.dump(res, f)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
    
