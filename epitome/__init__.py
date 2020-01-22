r"""
==============
epitome Module
==============
.. currentmodule:: epitome

epitome is a computational model for predicting ChIP-seq peaks in a new cell type
from chromatin accessibility and known ChIP-seq peaks from ENCODE. This module
also includes scripts for processing ENCODE peaks.

.. automodule:: epitome.models
.. automodule:: epitome.functions
.. automodule:: epitome.viz
.. automodule:: epitome.constants

"""
import os
from os.path import expanduser

__path__ = __import__('pkgutil').extend_path(__path__, __name__)

S3_DATA_PATH = 'https://epitome-data.s3-us-west-1.amazonaws.com/data.zip'


def GET_EPITOME_USER_PATH():
    return os.path.join(os.path.expanduser('~'), '.epitome')

# relative path to data
def GET_DATA_PATH():
    return os.path.join(GET_EPITOME_USER_PATH(),'data')

POSITIONS_FILE = "all.pos.bed.gz"
FEATURE_NAME_FILE = "feature_name"
REQUIRED_FILES = [POSITIONS_FILE,"train.npz","valid.npz", FEATURE_NAME_FILE,"test.npz"]
