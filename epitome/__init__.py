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

from os.path import expanduser

__path__ = __import__('pkgutil').extend_path(__path__, __name__)
