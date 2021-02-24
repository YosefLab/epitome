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
.. automodule:: epitome.dataset
.. automodule:: epitome.generators
.. automodule:: epitome.conversion

"""
import os
from os.path import expanduser

__path__ = __import__('pkgutil').extend_path(__path__, __name__)

# os env that should be set by user to explicitly set the data path
EPITOME_DATA_PATH_ENV = "EPITOME_DATA_PATH" # Must be an absolute path.
EPITOME_ASSEMBLY_ENV = "EPITOME_ASSEMBLY"
EPITOME_GENOME_ASSEMBLIES = ['hg19', 'test']

# data files required by epitome
# data.h5 contains data, row information (celltypes and targets) and
# column information (chr, start, binSize)
EPITOME_H5_FILE = "data.h5"
REQUIRED_FILES = [EPITOME_H5_FILE]

def GET_EPITOME_USER_PATH():
    return os.path.join(os.path.expanduser('~'), '.epitome')

def LIST_GENOME_ASSEMBLIES():
    return ", ".join(EPITOME_GENOME_ASSEMBLIES)

def GET_DATA_PATH():
    """
    Check if user has set env variable that specifies data path.
    Otherwise, use default location.

    Returns:
        location of epitome data with all required files
    """
    epitome_data_path  = os.environ.get(EPITOME_DATA_PATH_ENV)
    epitome_assembly = os.environ.get(EPITOME_ASSEMBLY_ENV)

    # Throw error if both ENV variables are set
    both_set = (epitome_data_path is not None) and (epitome_assembly is not None)
    assert not both_set, "Only specify either the %s env variable or %s env variable. Cannot define both." % (EPITOME_DATA_PATH_ENV, EPITOME_ASSEMBLY_ENV)

    # Return specified data path if env variable is set
    if (epitome_data_path is not None):
        return epitome_data_path
    else:
        # Default to the hg19 assembly if assembly and data path is not specified
        if (epitome_assembly is None):
            epitome_assembly = 'hg19'
            print("Warning: genome assembly %s env variable was not set. Defaulting genome assembly to %s." % (EPITOME_ASSEMBLY_ENV, epitome_assembly))
        epitome_data_dir_path = os.path.join(GET_EPITOME_USER_PATH(), 'data')
        return os.path.join(epitome_data_dir_path, epitome_assembly)
