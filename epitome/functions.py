r"""
================
Helper functions
================
.. currentmodule:: epitome.functions

.. autosummary::
  :toctree: _generate/

  download_and_unzip
  bed2Pyranges
  indices_for_weighted_resample
  get_radius_indices
"""

# imports
from epitome import *
import h5py
from scipy.io import savemat
import csv
import mimetypes

import pandas as pd
import collections
import numpy as np
import os
from collections import Counter
from itertools import groupby
from scipy.io import loadmat
from .constants import *
import scipy.sparse
import pyranges as pr
from sklearn.metrics import jaccard_score

import warnings
from operator import itemgetter
import urllib
import sys
import requests
import urllib
import tqdm
from zipfile import ZipFile
import gzip
import shutil

# to load in positions file
import multiprocessing

def download_and_unzip(url, dst):
    '''
    Downloads a url to local destination, unzips it and deletes zip.

    :param str url: url to download.
    :param str dst: local absolute path to download data to.
    '''
    if not os.path.exists(dst):
        os.makedirs(dst)

    dst = os.path.join(dst, os.path.basename(url))

    final_dst = dst.split('.zip')[0]

    # download data if it does not exist
    if not os.path.exists(final_dst):

        file_size = int(urllib.request.urlopen(url).info().get('Content-Length', -1))
        if os.path.exists(dst):
            first_byte = os.path.getsize(dst)
        else:
            first_byte = 0
        if first_byte < file_size:

            header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
            pbar = tqdm.tqdm(
                total=file_size, initial=first_byte,
                unit='B', unit_scale=True, desc="Dataset not found. Downloading Epitome data to %s..." % dst)
            req = requests.get(url, headers=header, stream=True)
            with(open(dst, 'ab')) as f:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(1024)
            pbar.close()

    if url.endswith('.zip'):

        # Extract zip data if it does not exist
        if not os.path.exists(final_dst):
            with ZipFile(dst, 'r') as zipObj:
               zipObj.extractall(os.path.dirname(dst))

            # delete old zip to free space
            os.remove(dst)

################### Parsing data from bed file ########################
def bed2Pyranges(bed_file):
    '''
    Loads bed file in as a pyranges object.
    Preserves ordering of bed lines by loading in as a pandas DF first.

    :param str bed_file: absolute path to bed file
    :return: indexed pyranges object
    :rtype: pyranges object
    '''

    # check to see whether there is a header
    # usually something of the form "chr start end"
    if mimetypes.guess_type(bed_file)[1] == 'gzip':

        with gzip.open(bed_file) as f:
            header = csv.Sniffer().has_header(f.read(1024).decode())

    else:
        with open(bed_file) as f:
            header = csv.Sniffer().has_header(f.read(1024))

    if not header:
        p = pd.read_csv(bed_file, sep='\t',header=None)[[0,1,2]]
    else:
        # skip header row
        p = pd.read_csv(bed_file, sep='\t',skiprows=1,header=None)[[0,1,2]]

    p['idx']=p.index
    p.columns = ['Chromosome', 'Start','End','idx']
    return pr.PyRanges(p, int64=True).sort()


def indices_for_weighted_resample(data, n,  matrix, cellmap, assaymap, weights = None):
    '''
    Selects n rows from data that have the greatest number of labels (can be weighted)
    Returns indices to these rows.

    :param numpy.matrix data: data matrix with shape (factors, records)
    :param int n: number or rows to sample
    :param numpy.matrix matrix: cell type by assay position matrix
    :param dict cellmap: dict of cells and row positions in matrix
    :param dict assaymap: dict of assays and column positions in matrix
    :param numpy.array weights: Optional vector of weights whos length = # factors (1 weight for each factor).
        The greater the weight, the more the positives for this factor matters.
    :return: numpy matrix of indices
    :rtype: numpy.matrix
    '''

    raise Exception("This function has not been modified to not use DNase")
    # only take rows that will be used in set
    # drop DNase from indices in assaymap first
    selected_assays = list(assaymap.values())[1:]
    indices = matrix[list(cellmap.values())][:,selected_assays].flatten()

    # set missing assay/cell combinations to -1
    t1 = data[indices, :]
    t1[np.where(indices < 0)[0],:] = 0

    # sum over each factor for each record
    sums = np.sum(np.reshape(t1, (len(selected_assays), len(cellmap), t1.shape[1])), axis=1)

    if (weights is not None):
        weights = np.reshape(weights, (weights.shape[0],1)) # reshape so multiply works
        probs = np.sum(sums * weights, axis = 0)
        probs = probs/np.sum(probs)
    else:
        # simple sum over recoreds. Weights records with more positive
        # samples higher for random sampling.
        probs = np.sum(sums, axis=0)
        probs = (probs)/np.sum(probs)

    # TODO assign equal probs to non-zero weights
    probs[probs != 0] = 1/probs[probs != 0].shape[0]

    radius = 20

    n = int(n / radius)
    data_count = data.shape[1]

    # sample by probabilities. not sorted.
    choice = np.random.choice(np.arange(0, data_count), n, p = probs)

    func_ = lambda x: np.arange(x - radius/2, x + radius/2)
    surrounding = np.unique(list(map(func_, choice)))
    return surrounding[(surrounding > 0) & (surrounding < data_count)].astype(int)


def get_radius_indices(radii, r, i, max_index):
    '''
    Gets indices for a given radius r in both directions from index i.
    Used in generator code to get indices in data for a given radius from
    genomic loci i.

    :param list radii: increasing list of integers indiciating radii
    :param int r: Index of which radii
    :param int i: center index to access data
    :param int max_index: max index which can be accessed

    :return: exclusive indices for this radius
    :rtype: numpy.array
    '''
    radius = radii[r]

    min_radius = max(0, i - radius)
    max_radius = min(i+radius+1, max_index)

    # do not featurize chromatin regions
    # that were considered in smaller radii
    if (r != 0):

        radius_range_1 = np.arange(min_radius, max(0, i - radii[r-1]+1))
        radius_range_2 = np.arange(i+radii[r-1], max_radius)

        radius_range = np.concatenate([radius_range_1, radius_range_2])
    else:

        radius_range = np.arange(min_radius, max_radius)

    return radius_range

def compute_casv(m1, m2, radii, indices= None):
    '''
    Computes CASV between two matrices. CASV indiciates how similar
    two binary matrices are to eachother. m1 and m2 should have the
    same number of rows and columns, where rows indicate regions and
    columns indicate the assays used to compute the casv (ie DNase-seq, H3K27ac)
    :param np.matrix m1: 2D or 3D numpy matrix 2D shape (nregions x (nassays x ncelltypes))
      where 2nd dimension is blocked by cells (i.e. cell1assay1, cell1assay2, cell2assay1, cell2assay2)
      OR 3D: (nregions x nassays x ncells)
    :param np.matrix m2: 3D numpy matrix shape (nregions x nassays x nsamples)
    :param radii: list of radii to access surrounding region
    :param indices: indices on 0th axis of m1 and m2 to compute casv for
    :return numpy matrix of size (len(indices) x CASV dimension x ncelltypes x ncells)
    '''

    if indices is None:
        indices = range(m1.shape[0])

    # if only one sample, extend m2 along 2nd axis
    if len(m2.shape) == 2:
        m2 = m2[:,:,None]

    # if needed, reshape m1 to put all assay/train cells on the last axis
    if len(m1.shape) == 3:
      ncells = m1.shape[-1]
      m1 = m1.reshape(m1.shape[0],m1.shape[1]*m1.shape[2])
    else:
      denom = 1 if m2.shape[1]==0 else m2.shape[1]
      ncells = int(m1.shape[-1]/denom)

    if m2.shape[1] == 0:
      # in this case, there is no CASV to compute, so we just return
      return np.zeros((len(indices),0, ncells,m2.shape[-1]))

    print(m1.shape, m2.shape)
    assert m1.shape[0] == m2.shape[0]
    # verify number of assays match
    assert m2.shape[1] == m1.shape[-1]/ncells
    # print('HERE')
    
#     set_trace()

    def f(i):
        
#         set_trace()
        # get indices for each radius in radii
        radius_ranges = list(map(lambda x: get_radius_indices(radii, x, i, m1.shape[0]), range(len(radii))))

        if len(radius_ranges) > 0:
            radius_indices = np.concatenate(radius_ranges)

            # data from known cell types (m1 portion)
            m1_slice = m1[radius_indices, :]
            m2_slice = np.repeat(m2[radius_indices, :, :],axis=1, repeats = ncells)
            

            # shape: radius size x (nassaysxncells) by nsamples
            pos = (m1_slice.T*m2_slice.T).T
#             agree = (m1_slice.T == m2_slice.T).T

            # split pos and agree arrays to create new dimension for ncells
            # the new dimension will be 4D: (radius x nassays x ncells x nsamples)
            pos = np.stack(np.split(pos, ncells, axis=1), axis=2)
#             agree = np.stack(np.split(agree, ncells, axis=1), axis=2)
            
            # get indices to split on. remove last because it is empty
            split_indices = np.cumsum([len(i) for i in radius_ranges])[:-1]
            # slice arrays by radii
            pos_arrays = np.split(pos, split_indices, axis= 0 )
#             agree_arrays = np.split(agree, split_indices, axis = 0)

            # average over the radius (0th axis)
            tmp1 = list(map(lambda x: np.average(x, axis = 0), pos_arrays)) # this line is problematic
            # final concatenation combines agree, nassays, and radii on the 0th axis
            # this axis is ordered by (1) pos/agree, then (2) radii, then (2) n assays.
            # See ordering example when there are 2 radii (r1, r2):
            # - pos: r1, nassays | pos: r2, nassays | agree: r1: nassays | agree: r1: nassays
            tmp = np.concatenate(tmp1, axis=0)
            return tmp
        else:
            # no radius, so no similarities. just an empty placeholder
            # shaped with the number of cells (last dim of m1)
            return np.zeros((0,ncells,m2.shape[-1]))

    # for every region of interest
    # TODO: maybe something more efficient?

    # set_trace()
    tmp = []
    for i in indices:
        tmp.append(f(i))
    
    return np.stack(tmp)
#     return np.stack([f(i