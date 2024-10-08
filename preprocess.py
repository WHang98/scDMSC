from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle, os, numbers

import numpy as np
import scipy as sp
import pandas as pd
import scanpy   as sc
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import scipy
from sklearn.preprocessing import LabelEncoder
from utils import *

#TODO: Fix this
class AnnSequence:
    def __init__(self, matrix, batch_size, sf=None):
        self.matrix = matrix
        if sf is None:
            self.size_factors = np.ones((self.matrix.shape[0], 1),
                                        dtype=np.float32)
        else:
            self.size_factors = sf
        self.batch_size = batch_size

    def __len__(self):
        return len(self.matrix) // self.batch_size

    def __getitem__(self, idx):
        batch = self.matrix[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_sf = self.size_factors[idx*self.batch_size:(idx+1)*self.batch_size]

        # return an (X, Y) pair
        return {'count': batch, 'size_factors': batch_sf}, batch


def read_dataset(adata, transpose=False, test_split=False, copy=False):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata

def clr_normalize_each_cell(adata):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)
    
    adata.raw = adata.copy()
    sc.pp.normalize_per_cell(adata)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.raw.X.A if scipy.sparse.issparse(adata.raw.X) else adata.raw.X)
    )
    return adata
    
def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
            if filter_min_counts:
                sc.pp.filter_genes(adata, min_counts=1)
                sc.pp.filter_cells(adata, min_counts=1)
            if size_factors or normalize_input or logtrans_input:
                adata.raw = adata.copy()
            else:
                adata.raw = adata
            if size_factors:
                 sc.pp.normalize_per_cell(adata)
                 adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
            else:
                adata.obs['size_factors'] = 1.0
            if logtrans_input:
               sc.pp.log1p(adata)
            if normalize_input:
               sc.pp.scale(adata)
            return adata

def read_genelist(filename):
    genelist = list(set(open(filename, 'rt').read().strip().split('\n')))
    assert len(genelist) > 0, 'No genes detected in genelist file'
    print('### Autoencoder: Subset of {} genes will be denoised.'.format(len(genelist)))

    return genelist

def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')
def read_pickle(inputfile):
    return pickle.load(open(inputfile, "rb"))

def load_data():
    print('loading data!')
    data_mat = h5py.File(r'spector.h5')
    x1 = np.array(data_mat['X1'])
    x2 = np.array(data_mat['X2'])

    y = np.array(data_mat['Y'])
    data_mat.close()

    #Gene filter
    importantGenes = geneSelection(x1, n=2000, plot=False)           #基因选择的个数
    x1 = x1[:, importantGenes]

    # importantGenes = geneSelection(x2, n=4000, plot=False)           #染色质选择的个数
    # x2 = x2[:, importantGenes]

    x = np.array(np.concatenate([x1,x2],axis=1))       

    adata = sc.AnnData(x)
    adata1 = sc.AnnData(x1)
    adata2 = sc.AnnData(x2)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata1 = read_dataset(adata1,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata1 = normalize(adata1,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    adata2 = sc.AnnData(x2)
    adata2.obs['Group'] = y
    adata2 = read_dataset(adata2,
                     transpose=False,
                     test_split=False,
                     copy=True)
    
    adata2 = normalize(adata2,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    label=np.array(y)
    Y=pd.DataFrame(label)
    Y = Y.dropna()
    #keys = y.keys()
    Y = Y[Y.keys()].apply(LabelEncoder().fit_transform)
    label = np.array(Y,dtype=int)
    label=label.reshape(-1,)
    Y=np.array(label).astype(int)

    data_mat.close()

    print(len(set(y)))
    print(y)
    return adata1.X,adata2.X,adata.X,y