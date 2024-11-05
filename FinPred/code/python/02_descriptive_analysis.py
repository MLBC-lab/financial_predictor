#!/bin/python

"""
Current Step:
o Find the highly missing features (more than some threshold)
o Find the features' types: categorical or continues
o Find the highly missing targets (more than some threshold)
o Find the targests' types: categorical or continues
o Find the balance of categorical targets

Last step:
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

# Globals
#
br = breakpoint
TARGETS_IFILE = '../../data/targets.pckl'
FEATS_IFILE = '../../data/feats.pckl'
IDENTIFIERS = ['ticker', 'year', 'gvkey']
MISS_THRESHOLD = 0.2
CATEGORICAL = 40

def main(argv):
    with open(FEATS_IFILE, mode='rb') as f:
        data = pickle.load(f)
    with open(TARGETS_IFILE, mode='rb') as f:
        trgs = pickle.load(f)
    feat_cols = [col for col in data.columns
                 if not ((col in IDENTIFIERS) | (col in trgs))]
    miss_feat = [ft for ft in feat_cols
                  if (data[ft].isna().sum() / data.shape[0]) > MISS_THRESHOLD]
    cat_feat = [ft for ft in feat_cols
                 if len(data[ft].unique()) < CATEGORICAL]
    miss_targ = [tg for tg in trgs
                 if data[tg].isna().sum() / data.shape[0] > MISS_THRESHOLD]
    cat_targ = [tg for tg in trgs if len(data[tg].unique()) < CATEGORICAL]
    bal_cat_targ = {
        tg: np.unique(data[tg], return_counts=True)[1] / data.shape[0]
        for tg in cat_targ}
    # Some info
    print(f'Number of samples: {data.shape[0]}')
    print(f'Number of features: {len(feat_cols)}')
    print(f'Number of missed features ({MISS_THRESHOLD}): {len(miss_feat)}')
    print(f'Number of categorical features (less than {CATEGORICAL} category): '
          f'{len(cat_feat)}')
    print(f'Number of targets: {len(trgs)}')
    print(f'Number of missed targets ({MISS_THRESHOLD}): {len(miss_targ)}')
    print(f'Number of categorical targets (less than {CATEGORICAL} category): '
          f'{len(cat_targ)}')
    print('Balance of categorical targets:')
    for tg in bal_cat_targ:
        print(tg, np.round(bal_cat_targ[tg], 3))
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
    
