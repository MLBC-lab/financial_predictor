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
TARGETS_IFILE = '../../data/targets.pckl'
FEATS_IFILE = '../../data/feats.pckl'
# RESULT_OFILE = '../../output/cross_validate_result.pckl'
RESULT_ODIR = '../../output/targets_results/'
PLOT_ODIR = '../../output/plots/'
IDENTIFIERS = ['ticker', 'year', 'gvkey']
MISS_THRESHOLD = 0.1
CATEGORICAL = 40
NPCA = 10
NPS = 4

def get_pipeline():
    # clf = sklearn.kernel_ridge.KernelRidge()
    clf = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    ppl = sklearn.pipeline.Pipeline(steps = [
        ('scaler',
         sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)),
        ('pca', sklearn.decomposition.PCA(n_components=NPCA)),
        ('clf', clf)])
    return ppl

def cross_validate(data, feat_cols, tg):
    """
    data: Pandas' data frame
    feat_cols: a list of feature names
    tg: the current target that should be predicted
    """
    targ_index = data[tg].notna()
    targ = data[targ_index][tg].values
    targ = sklearn.preprocessing.StandardScaler(
        with_mean=True, with_std=True).fit_transform(
            targ.reshape(-1, 1)).flatten()
    # remove features with more than MISS_THRESHOLD% missed values
    feat = np.array(
        [data[targ_index][ft] for ft in feat_cols
         if (data[targ_index][ft].isna().sum() / targ_index.sum())
         < MISS_THRESHOLD]).T
    # Fill missing values with the mean of features
    impute = sklearn.impute.SimpleImputer(missing_values=np.nan,
                                          strategy='mean')
    feat = impute.fit_transform(feat)
    ppl = get_pipeline()
    result = sklearn.model_selection.cross_validate(
        ppl, feat, targ,
        scoring='neg_mean_squared_error',
        # scoring='r2',
        return_train_score=True, return_estimator=True, return_indices=True)
    print(f'Target: {tg}, Len: {targ.shape[0]}, feat: {feat.shape[1]}, '
          f'Train: {-np.round(result["train_score"].mean(), 2)}, '
          f'Test: {-np.round(result["test_score"].mean(), 2)}',
          flush=True)
    return result

def train_validate(data, feat_cols, tg):
    # split data into train and test based on tickers
    train_tickers, test_tickers = 
    
    result = cross_validate(data_feat_cols, tg)
    # save the result
    fname = os.path.join(RESULT_ODIR, tg + '.pckl')
    with open(fname, mode='wb') as f:
        pickle.dump(result, f)
    # get best model index on the test set
    best_index = result['test_score'].argmax()
    best_estimator = result['estimator'][best_index]
    # Plot some time-series for companies
    tickers = data[targ_index]['ticker'].unique()
    counter = 0
    for tick in tickers:
        x = [f'{d["ticker"]}_{d.year}'
             for _, d in
             data[targ_index][data[targ_index]['ticker'] == tick].iterrows()]
        y = best_estimator.predict(feat[counter:counter + len(x)])
        target_trace = go.Scatter(x=np.arange(len(y)), y=y,
                                  name='target',)
                                  # labels=dict(x='ticker_year', y=tg))
        predict_trace = go.Scatter(x=np.arange(len(y)),
                                   y=targ[counter:counter+len(x)],
                                   name='predict', )
        fig = plotly.subplots.make_subplots()
        fig.add_trace(target_trace)
        fig.add_trace(predict_trace)
        fig.update_layout(xaxis=dict(tickmode='array',
                                     tickvals=np.arange(len(y)),
                                     ticktext=x))
        fig.layout.font = dict(size=10)
        fig.write_html(os.path.join(PLOT_ODIR, tick + '.html'))
        br()
        counter += len(x)
    # # Plot predicted errors
    # train_indices, test_indices = (result['indices']['train'][best_index],
    #                                result['indices']['test'][best_index])
    # train_feat, train_targ = feat[train_indices, :], targ[train_indices]
    # test_feat, test_targ = feat[test_indices, :], targ[test_indices]
    # for setn, sset, tg in [('train', train_feat, train_targ),
    #                        ('test', test_feat, test_targ)]:
    #     fig, axis = plt.subplots(1, figsize=(10, 10))
    #     ped = sklearn.metrics.PredictionErrorDisplay.from_estimator(
    #         best_estimator, sset, tg, ax=axis)
    #     fig.savefig(f'{setn}.png')
    return True
        
def main(argv):
    with open(FEATS_IFILE, mode='rb') as f:
        data = pickle.load(f)
    with open(TARGETS_IFILE, mode='rb') as f:
        trgs = pickle.load(f)
    feat_cols = [col for col in data.columns
                 if not ((col in IDENTIFIERS) | (col in trgs))]
    # find targets and available features
    #
    # Single Processing
    for tg in trgs[1:2]:
        train_validate(data, feat_cols, tg)
    # Multiprocessing
    # with mlps.pool.Pool(NPS) as pool:
    #     results = pool.starmap(cross_validate,
    #                            [(data, feat_cols, tg) for tg in trgs[:3]])
    # # Save the results
    # print(f'Saving the results as {RESULT_OFILE} ...')
    # with open(RESULT_OFILE, mode='wb') as f:
    #     pickle.dump(results, f)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
    
