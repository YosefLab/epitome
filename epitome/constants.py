# imports
from enum import Enum
import numpy as np
import os

REGIONS_FILENAME = 'all.pos.bed.gz'

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
    TRAIN = 1    # TRAINING
    VALID = 2    # VALIDATION during training
    TEST = 3     # TEST held out test set
    RUNTIME = 4  # Using the model at runtime. No truth, just predictions
