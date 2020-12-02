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
    data_path = os.path.join(GET_EPITOME_USER_PATH(),'data')
    assemblies = [os.path.join(data_path, item) if os.path.isdir(item) for item in os.listdir(data_path)]
    if (len(assemblies) == 0):
        # Download "hg19" genome or throw error ?
        # check env variable, if it's set download that
        # throw error that it needs to be set
    elif (len(assemblies) == 1):
        return assemblies[0]
    else:
        # Throw error to specify assembly in env variable

def LIST_GENOMES():
    return ", ".join(GENOME_ASSEMBLIES)

def GET_GENOME(reference_genome=None, genome_url=None):
    if reference_genome in GENOME_ASSEMBLIES:
        # Call data/download_encode.py 
    elif (reference_genome not in GENOME_ASSEMBLIES and genome_url is None):
        return "Please provid a url to download %s" % reference_genome
    elif (reference_genome is None):
        return "Please specify a valid reference genome from %s or a different genome with a valid url to download it from." % LIST_GENOMES()
    else:
        return "Please specify a valid reference genome from %s or a different genome with a valid url to download it from." % LIST_GENOMES()

POSITIONS_FILE = "all.pos.bed.gz"
FEATURE_NAME_FILE = "feature_name"
REQUIRED_FILES = [POSITIONS_FILE,"train.npz","valid.npz", FEATURE_NAME_FILE,"test.npz"]
GENOME_ASSEMBLIES = set(['hg19','mm10','GRCh38'])
