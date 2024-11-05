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
import copy
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
e = lambda: os._exit(0)
EPSILON = 1e-6
NPS = 8
IS_SCALE, IS_PCA = False, False
DATA_IFILE = '../../data/divest_CSI.pckl'
# TARGETS_IFILE = '../../data/targets.pckl'
FEATS_IFILE = '../../data/feats.pckl'
TRAIN_TEST_IFILE = '../../data/train_test.pckl'
RESULT_OFILE = '../../output/cross_validate_result_scale_no_pca_no.pckl'
RESULT_DF_OFNAME = '../../output/cross_validate_result_scale_no_pca_no.xlsx'
# RESULT_ODIR = '../../output/targets_results/'
SCALERS = [sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1)),
           sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)]
PCAS = [sklearn.decomposition.PCA(n_components=10),
        sklearn.decomposition.PCA(n_components=40),
        sklearn.decomposition.PCA(n_components=200)]
ESTIMATORS = [sklearn.neighbors.KNeighborsRegressor(n_neighbors=5),
              sklearn.ensemble.RandomForestRegressor(n_estimators=20),
              sklearn.ensemble.RandomForestRegressor(n_estimators=100),
              sklearn.svm.SVR()]

def get_pipelines(scale=False, pca=False):
    ppls = []
    ## Scale: No, PCA: No
    if not (scale | pca):
        [ppls.append(sklearn.pipeline.Pipeline([('clf', clf)]))
         for clf in ESTIMATORS]
    ## Scale: Yes, PCA: No
    if scale & (not pca):
        [ppls.append(sklearn.pipeline.Pipeline([('scale', sc), ('clf', clf)]))
         for sc in SCALERS for clf in ESTIMATORS]
    ## Scale: Yes, PCA: Yes
    if scale & pca:
        [ppls.append(sklearn.pipeline.Pipeline([('scale', sc), ('pca', pca),
                                                ('clf', clf)]))
         for sc in SCALERS for pca in PCAS for clf in ESTIMATORS]
    return ppls

def clean_feats_targets(feat, targ):
    # rescaling the targets
    # targ = sklearn.preprocessing.StandardScaler(
    #     with_mean=True, with_std=True).fit_transform(
    #         targ.reshape(-1, 1)).flatten()
    # Fill missing values with the mean of features
    impute = sklearn.impute.SimpleImputer(missing_values=np.nan,
                                          strategy='mean')
    feat = impute.fit_transform(feat)
    # # Rescaling of features and targets
    # # feat = sklearn.preprocessing.StandardScaler().fit_transform(feat)
    # targ = sklearn.preprocessing.StandardScaler().fit_transform(
    #     targ.reshape((-1, 1))).flatten()
    # # targ = sklearn.preprocessing.MinMaxScaler(
    # #     feature_range=(-1, 1)).fit_transform(targ.reshape((-1, 1))).flatten()
    return feat, targ

def cross_validate(feat, targ, ppl):
    """
    data: Pandas' data frame
    feat_cols: a list of feature names
    tg: the current target that should be predicted
    """
    feat, targ = clean_feats_targets(feat, targ)
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
    # select the best model
    best_score = float('-Inf')
    ppls = get_pipelines(scale=IS_SCALE, pca=IS_PCA)
    for ppl in ppls:
        # training by cross validation
        ppl = copy.deepcopy(ppl)
        result = cross_validate(tr_feat, tr_targ, ppl)
        # get best model index on the test set
        score = result['test_score'].max()
        if score > best_score:
            best_score = score
            best_index = result['test_score'].argmax()
            best_estimator = result['estimator'][best_index]
            # print(best_estimator, best_score)
    # get best estimator's score on test data
    tr_score = validate(best_estimator, tr_feat, tr_targ)
    ts_score = validate(best_estimator, ts_feat, ts_targ)
    # Normalized MSE
    tr_score_norm = tr_score / (tr_targ.var() + EPSILON)
    ts_score_norm = ts_score / (ts_targ.var() + EPSILON)
    # Plot some time-series for companies
    # plot_ts(train_data)
    res = {'estimator': best_estimator,
           'tr_mse': tr_score, 'ts_mse': ts_score,
           'tr_mse_norm': tr_score_norm, 'ts_mse_norm': ts_score_norm
    }
    return res

def process(data, trts, tg):
    t0 = time.time()
    tg_data = data[data[tg].notna()]
    feat_names = trts[tg]['feat_names']
    tr_companies = trts[tg]['train']
    ts_companies = trts[tg]['test']
    tr_data = tg_data[tg_data['ticker'].isin(tr_companies)]
    ts_data = tg_data[tg_data['ticker'].isin(ts_companies)]
    tr_targ, tr_feat = tr_data[tg].values, tr_data[feat_names].values
    ts_targ, ts_feat = ts_data[tg].values, ts_data[feat_names].values
    if tg == 'mva_t3_1':
        tr_targ = sklearn.preprocessing.StandardScaler().fit_transform(
            tr_targ.reshape((-1, 1))).flatten()
        ts_targ = sklearn.preprocessing.StandardScaler().fit_transform(
            ts_targ.reshape((-1, 1))).flatten()
    result = {'tg': tg, 'nfeat': len(feat_names), 'nsample': tg_data.shape[0]}
    res = train_validate(tr_feat, tr_targ, ts_feat, ts_targ)
    result.update(res)
    print(f'{tg}, Dur: {time.time() - t0:.0f} s, '
          f'({tg_data.shape[0]}, {len(feat_names)}), '
          f'TrMSE: {np.round(result["tr_mse"], 2)}, '
          f'TsMSE: {np.round(result["ts_mse"], 2)}\n'
          f'{result["estimator"]}', flush=True)
    return result

def main(argv):
    with open(DATA_IFILE, mode='rb') as f:
        data = pickle.load(f)
    # with open(TARGETS_IFILE, mode='rb') as f:
    #     trgs = pickle.load(f)
    # feat_cols = [col for col in data.columns
    #              if not ((col in IDENTIFIERS) | (col in trgs))]
    with open(TRAIN_TEST_IFILE, mode='rb') as f:
        trts = pickle.load(f)
    # Print parameters
    res = []
    # # Single Processing
    # for tg in list(trts.keys())[:3]:
    #     res.append(process(data, trts, tg))
    # Multiprocessing
    trts_keys = sorted(list(trts.keys()))
    with mlps.pool.Pool(NPS) as pool:
        res = pool.starmap(process, [(data, trts, tg) for tg in trts_keys])
    # Save the results
    print(f'Saving the results as {RESULT_OFILE} & {RESULT_DF_OFNAME} ...')
    with open(RESULT_OFILE, mode='wb') as f:
        pickle.dump(res, f)
    df = pandas.DataFrame.from_records(res)
    df.to_excel(RESULT_DF_OFNAME)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
    
