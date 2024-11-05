#!/bin/python

"""
Current Step:
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
import pandas

# Globals
#
br = breakpoint
DATA_IFILE = '../../data/divest_CSI.xlsx'
TARGETS_IFILE = '../../data/targets_Reduced_28June2024.xlsx'
FEATS_IFILE = '../../data/features_by_group_20240731.xlsx'
DATA_OFILE = '../../data/divest_CSI.pckl'
TARGETS_OFILE = '../../data/targets.pckl'
# FEATS_EXCEL_OFILE = '../../data/feats.xlsx'
FEATS_OFILE = '../../data/feats.pckl'
IDENTIFIERS = ['ticker', 'year', 'gvkey']

def read_pickle(fname, mode='r'):
    with open(fname, mode=mode) as f:
        data = pickle.load(f)
    return data
def write_pickle(data, fname, mode='w'):
    with open(fname, mode=mode) as f:
        pickle.dump(data, f)
    return True

def main(argv):
    # read data
    if os.path.isfile(DATA_OFILE):
        data = read_pickle(DATA_OFILE, mode='rb')
    else:
        data = pandas.read_excel(DATA_IFILE)
        write_pickle(data, DATA_OFILE, mode='wb')
    # read targets names
    if os.path.isfile(TARGETS_OFILE):
        trgs = read_pickle(TARGETS_OFILE, mode='rb')
    else:
        trgs = pandas.read_excel(TARGETS_IFILE, header=None)
        trgs = trgs[0].values.tolist()
        write_pickle(trgs, TARGETS_OFILE, mode='wb')
    # read and arrange feature names and category
    if os.path.isfile(FEATS_OFILE):
        feat_group = read_pickle(FEATS_OFILE, mode='rb')
    else:
        feat_group = pandas.read_excel(FEATS_IFILE)
        group = feat_group.columns.tolist()
        feat_group = {gr: feat_group[gr][feat_group[gr].notna()].tolist()
                      for gr in group}
        feat_group['all'] = sum(list(feat_group.values()), [])
    # feat_cols = [col for col in data.columns
    #              if not ((col in IDENTIFIERS) | (col in trgs) | (col in mtrgs))]
    # reorder the columns
    # odata = pandas.concat((data[IDENTIFIERS], data[feat_cols], data[mtrgs]),
    #                       axis=1)
    print('Saving the output feature file ...')
    write_pickle(feat_group, FEATS_OFILE, mode='wb')
    # odata.to_excel(FEATS_EXCEL_OFILE)
    return True

if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f'Finished in {time.time() - t0:.0f} seconds.')
    
