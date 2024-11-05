#!/bin/python

"""
Current Step:
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

# Globals
#
br = breakpoint
DATA_IFILE = '../../data/divest_CSI.pckl'
FEATS_IFILE = '../../data/feats.pckl'
TRAIN_TEST_IFILE = '../../data/train_test.pckl'
RESULT_IFILE = '../../output/cross_validate_result_scale_no_pca_no.pckl'
FEAT_IMPORT_OFILE = '../../output/features_importances_mva_t3_1.pckl'

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
    targ = sklearn.preprocessing.StandardScaler().fit_transform(
        targ.reshape((-1, 1))).flatten()
    feat = impute.fit_transform(feat)
    return feat, targ

def main(argv):
    with open(DATA_IFILE, mode='rb') as f:
        data = pickle.load(f)
    with open(FEATS_IFILE, mode='rb') as f:
        feat_group = pickle.load(f)
    with open(TRAIN_TEST_IFILE, mode='rb') as f:
        trts = pickle.load(f)
    with open(RESULT_IFILE, mode='rb') as f:
        results = pickle.load(f)
    features_importances = dict()
    for res in results:
        tg, est, tr_mse, ts_mse = (res['tg'], res['estimator'],
                                   res['tr_mse'], res['ts_mse'])
        if tg != 'mva_t3_1':
            continue
        features_importances[tg] = dict()
        print(f'{tg}, tr_mse: {tr_mse:.3f}, ts_mse: {ts_mse:.3f}')
        # # make output directories
        # tr_plots_tg_odir = os.path.join(tr_plots_odir, tg)
        # ts_plots_tg_odir = os.path.join(ts_plots_odir, tg)
        # os.makedirs(tr_plots_tg_odir, exist_ok=True)
        # os.makedirs(ts_plots_tg_odir, exist_ok=True)
        # construct data
        tg_data = data[data[tg].notna()]
        feat_names = trts[tg]['feat_names']
        tr_companies = trts[tg]['train']
        ts_companies = trts[tg]['test']
        tr_data = tg_data[tg_data['ticker'].isin(tr_companies)]
        ts_data = tg_data[tg_data['ticker'].isin(ts_companies)]
        tr_targ, tr_feat = tr_data[tg].values, tr_data[feat_names].values
        ts_targ, ts_feat = ts_data[tg].values, ts_data[feat_names].values
        # combine
        targ, feat = (np.hstack((tr_targ, ts_targ)),
                      np.vstack((tr_feat, ts_feat)))
        feat, targ = clean_feats_targets(feat, targ)
        pred = est.predict(feat)
        mse = performance(targ, pred)
        pi = sklearn.inspection.permutation_importance(
            est, feat, targ, scoring='neg_mean_squared_error',
            n_repeats=5)
        feat_indices = pi.importances_mean.argsort()[::-1].tolist()
        feat_importance = [feat_names[c] for c in feat_indices]
        features_importances[tg]['feat_importance'] = pi
        print(f'top 10 important features (descending):\n'
              f'{feat_importance[:10]}')
        ranks = {gr: None for gr in feat_group if gr != 'all'}
        mean_score = {gr: None for gr in feat_group if gr != 'all'}
        max_score = {gr: None for gr in feat_group if gr != 'all'}
        for gr in ranks:
            ranks[gr] = {ft: feat_indices.index(feat_names.index(ft))
                         for ft in feat_group[gr] if ft in feat_names}
            score = [pi.importances_mean[feat_names.index(ft)]
                     for ft in feat_group[gr] if ft in feat_names]
            mean_score[gr], max_score[gr] = np.mean(score), max(score)            
            print(f'{gr}: {ranks[gr]}')
        # # sort scores
        # mean_score = {gr: mean_score[gr]
        #               for gr in sorted(mean_score, reverse=True,
        #                                key=lambda k: mean_score[k])}
        # max_score = {gr: max_score[gr]
        #              for gr in sorted(max_score, reverse=True,
        #                               key=lambda k: max_score[k])}
        features_importances[tg]['mean_score'] = mean_score
        features_importances[tg]['max_score'] = max_score
        print(mean_score)
        print(max_score)
    print(f'Saving feature importances in {FEAT_IMPORT_OFILE} ... ')
    with open(FEAT_IMPORT_OFILE, mode='wb') as f:
        pickle.dump(features_importances, f)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
    
