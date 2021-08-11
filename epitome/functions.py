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

################## LOADING DATA ######################

def download_and_unzip(url, dst):
    '''
    Downloads a url to local destination, unzips it and deletes zip.

    :param str url: url to download.
    :param str dst: local absolute path to download data to.
    '''
    if not os.path.exists(dst):
        os.makedirs(dst)

    input_dst = dst
    dst = os.path.join(dst, os.path.basename(url))

    final_dst = dst.split('.zip')[0]

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
        if (not os.path.exists(final_dst)) or (len(os.listdir(final_dst)) == 0):
            with ZipFile(dst, 'r') as zipObj:
               zipObj.extractall(os.path.dirname(dst))
            # delete old zip to free space
            os.remove(dst)
            # delete nested directory
            for f in os.listdir(final_dst):
                os.rename(os.path.join(final_dst, f), os.path.join(input_dst, f))
            os.rmdir(final_dst)

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
