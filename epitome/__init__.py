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


def GET_EPITOME_USER_PATH():
    return os.path.join(os.path.expanduser('~'), '.epitome')

def GET_DATA_PATH():
	"""
	Check if user has set env variable that specifies data path.
	Otherwise, use default location.

	Returns:
		location of epitome data with all required files
	"""
	if os.environ.get("EPITOME_DATA_PATH") is not None:
		return os.environ["EPITOME_DATA_PATH"]
	else:
		return os.path.join(GET_EPITOME_USER_PATH(),'data','hg19')
