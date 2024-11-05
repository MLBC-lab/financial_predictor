#!/bin/python

"""
Current Step:
o Make a new hierarchy and save it into two different datasets:
    Target
        Feature_Names
        Train
            Company
        Test
            Company

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
import pandas
import sklearn
import sklearn.model_selection


# Globals
#
br = breakpoint
DATA_IFILE = '../../data/divest_CSI.pckl'
TARGETS_IFILE = '../../data/targets.pckl'
FEATS_IFILE = '../../data/feats.pckl'
TRAIN_TEST_OFILE = '../../data/train_test.pckl'
# IDENTIFIERS = ['ticker', 'year', 'gvkey']
TEST_RATIO = 0.2
MISS_THRESHOLD = 0.5

def main(argv):
    with open(DATA_IFILE, mode='rb') as f:
        data = pickle.load(f)
    with open(TARGETS_IFILE, mode='rb') as f:
        trgs = pickle.load(f)
    with open(FEATS_IFILE, mode='rb') as f:
        feat_group = pickle.load(f)
    feat_cols = feat_group['all']
    hier = dict()
    for tgc, tg in enumerate(trgs):
        # remove features with more than MISS_THRESHOLD% missed values
        tg_bool = data[tg].notna()
        tg_data = data[tg_bool]
        ntg = tg_bool.sum()
        # feat_names = [col for col in feat_cols
        #               if (tg_data[col].isna().sum()/targ_bool.sum())
        #               < MISS_THRESHOLD]
        feat_names = (np.array(feat_cols)[
            (tg_data[feat_cols].isna().sum(axis=0)/ntg)
            < MISS_THRESHOLD]).tolist()
        # hier[tg]['feat_names'] = feat_names
        companies = data[tg_bool]['ticker'].unique()
        train_comp, test_comp = sklearn.model_selection.train_test_split(
            companies, test_size=TEST_RATIO, shuffle=True)
        train_comp, test_comp = train_comp.tolist(), test_comp.tolist()
        hier[tg] = {
            'feat_names': feat_names, 'train': train_comp, 'test': test_comp}
        # # Populate train and test dictionaries
        # hier[tg]['train'] = dict()
        # for comp in train_comp:
        #     tg_comp_data = tg_data[tg_data['ticker'] == comp]
        #     tg_date = tg_comp_data.year
        #     tg_feat = tg_comp_data[feat_names].values
        #     hier[tg]['train'][comp] = {'date': tg_date, 'feat': tg_feat}
        # hier[tg]['test'] = dict()
        # for comp in test_comp:
        #     tg_comp_data = tg_data[tg_data['ticker'] == comp]
        #     tg_date = tg_comp_data.year
        #     tg_feat = tg_comp_data[feat_names].values
        #     hier[tg]['test'][comp] = {'date': tg_date, 'feat': tg_feat}
        print(f'{tgc+1} / {len(trgs)}', end='\r')
    print()
    with open(TRAIN_TEST_OFILE, mode='wb') as f:
        pickle.dump(hier, f)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
    
