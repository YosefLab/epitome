r"""
=========
Constants
=========
.. currentmodule:: epitome.constants

.. autosummary::
  :toctree: _generate/

  Dataset
"""

# imports
from enum import Enum
import numpy as np
import os

######################################################
################### CONSTANTS ########################
######################################################
class Features(Enum):
    MASK_IDX = 0
    FEATURE_IDX = 1

class Label(Enum):
    IMPUTED_UNBOUND = -3
    IMPUTED_BOUND = -2
    UNK = -1
    UNBOUND = 0
    BOUND = 1

class Dataset(Enum):
    """ Enumeration determining train, valid, test or runtime.
    """


    TRAIN = 1    # TRAINING
    r"""
    Training mode: Allows subsampling of 0s.
    """


    VALID = 2    # VALIDATION during training
    r"""
    Validation mode: Allows validation on training cell types.
    """


    TEST = 3     # TEST held out test set
    r"""
    Test mode: Disables subsampling of 0s.
    """

    RUNTIME = 4  # Using the model at runtime. No truth, just predictions
    r"""
    Runtime mode: Allows a new cell type to be predicted on.
    """
