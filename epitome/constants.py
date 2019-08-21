# imports
from enum import Enum
import numpy as np
import os


######################################################
##################### Files ##########################
######################################################
this_dir, this_filename = os.path.split(__file__)

DEEPSEA_ALLTFS_BEDFILE = os.path.join(this_dir, "..", "data", "DEEPSEA_DATA", "allTFs.pos.bed")
DEEPSEA_FEATURE_NAME_FILE = os.path.join(this_dir, "..", "data", "DEEPSEA_DATA", "feature_name")

EPITOME_ALLTFS_BEDFILE = os.path.join(this_dir, "..", "data", "EPITOME_DATA", "all.pos.bed")
EPITOME_FEATURE_NAME_FILE = os.path.join(this_dir, "..", "data", "EPITOME_DATA", "feature_name")

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
    

# these were the default assays (20) and cells(11) run in the original model.
# This is just to keep consistency. We can remove these, eventually.
DEFAULT_ASSAYS =  ['DNase', 'CTCF', 'Pol2', 'YY1', 'p300', 'TAF1', 'Pol2-4H8', 'c-Myc', 'Rad21', 'Max', 'NRSF', 'GABP', 'EZH2', 'CEBPB', 'c-Jun', 'ZBTB33', 'USF2', 'USF-1', 'TBP', 'RFX5']
DEFAULT_CELLS = ['K562', 'GM12878', 'H1-hESC', 'HepG2', 'HeLa-S3', 'A549', 'HUVEC', 'GM12891', 'MCF-7', 'GM12892', 'HCT-116']
    
# During validation, these assays performed > 0.7
OPTIMAL_ASSAYS = ["CTCF", "Pol2", "YY1", "Pol2-4H8", "p300", "c-Myc", "TAF1", "Max", "Rad21", "NRSF", "GABP", "CEBPB", "c-Jun", "ZBTB33", "USF2", "USF-1", "Sin3Ak-20", "RFX5", "Nrf1", "Mxi1", "JunD", "CHD2", "ATF3", "c-Fos", "Znf143", "TR4", "SP1", "SMC3", "SIX5", "Pol2(b)", "MafK", "MAZ", "ELF1", "COREST", "TEAD4", "SP2", "NF-YA", "HDAC2", "GTF2F1", "ETS1", "ELK1", "E2F4", "CHD1", "TBLR1", "TAF7", "RBBP5", "MafF", "MEF2A", "E2F6", "Brg1", "Bach1", "BCLAF1", "eGFP-JunD", "eGFP-JunB", "eGFP-GATA2", "eGFP-FOS", "UBF", "TRIM28", "THAP1", "SETDB1", "SAP30", "PHF8", "HDAC1", "GTF2B", "GATA-1", "CBX3"]


# AM 8/19/2019 TODO RM!

# # Regions in allpos.bed file that DeepSEA uses for training.
# # Note that they are contiguous, so the total regions used 
# # starts at _TRAIN_REGIONS[0] and ends at _TEST_REGIONS[1]+1
# DEEPSEA_TRAIN_REGIONS = [0, 2200000-1]
# DEEPSEA_VALID_REGIONS = [2200000, 2204000-1]
# # 227512 for test (chr8 and chr9) for 227512 total rows
# DEEPSEA_TEST_REGIONS  = [2309367, 2536878] # chr 8 and 9

# def N_TOTAL_REGIONS():
#     """
#     Get the total number of genomic regions in train, valid and test.
#     Should = 2431512
#     """
#     return (TEST_REGIONS[1]-TEST_REGIONS[0] + 1) +(VALID_REGIONS[1] + 1)
