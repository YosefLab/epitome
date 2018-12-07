# imports
from enum import Enum

######################################################
################### CONSTANTS ########################
######################################################
class Dataset(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3

# these were the default assays (20) and cells(11) run in the original model.
# This is just to keep consistency. We can remove these, eventually.
DEFAULT_ASSAYS =  ['DNase', 'CTCF', 'Pol2', 'YY1', 'p300', 'TAF1', 'Pol2-4H8', 'c-Myc', 'Rad21', 'Max', 'NRSF', 'GABP', 'EZH2', 'CEBPB', 'c-Jun', 'ZBTB33', 'USF2', 'USF-1', 'TBP', 'RFX5']
DEFAULT_CELLS = ['K562', 'GM12878', 'H1-hESC', 'HepG2', 'HeLa-S3', 'A549', 'HUVEC', 'GM12891', 'MCF-7', 'GM12892', 'HCT-116']
    
# During validation, these assays performed > 0.7
OPTIMAL_ASSAYS = ["CTCF", "Pol2", "YY1", "Pol2-4H8", "p300", "c-Myc", "TAF1", "Max", "Rad21", "NRSF", "GABP", "CEBPB", "c-Jun", "ZBTB33", "USF2", "USF-1", "Sin3Ak-20", "RFX5", "Nrf1", "Mxi1", "JunD", "CHD2", "ATF3", "c-Fos", "Znf143", "TR4", "SP1", "SMC3", "SIX5", "Pol2(b)", "MafK", "MAZ", "ELF1", "COREST", "TEAD4", "SP2", "NF-YA", "HDAC2", "GTF2F1", "ETS1", "ELK1", "E2F4", "CHD1", "TBLR1", "TAF7", "RBBP5", "MafF", "MEF2A", "E2F6", "Brg1", "Bach1", "BCLAF1", "eGFP-JunD", "eGFP-JunB", "eGFP-GATA2", "eGFP-FOS", "UBF", "TRIM28", "THAP1", "SETDB1", "SAP30", "PHF8", "HDAC1", "GTF2B", "GATA-1", "CBX3"]

# Regions in allpos.bed file that DeepSEA uses for training.
# Note that they are contiguous, so the total regions used 
# starts at _TRAIN_REGIONS[0] and ends at _TEST_REGIONS[1]+1
_TRAIN_REGIONS = [0, 2200000-1]
_VALID_REGIONS = [2200000, 2204000-1]
# 227512 for test (chr8 and chr9) for 227512 total rows
_TEST_REGIONS  = [2204000, 2204000 + 227512-1] # only up to chrs 8 and 9


    
# Modified regions in allpos.bed file that DeepSEA uses for training.
# We modified these because the validation set was too small.
_MODIFIED_TRAIN_REGIONS = np.r_[0:2200000,2400000:4200000]

# regions that should be taken from the train set for the validation set
_MODIFIED_VALID_FROM_TRAIN_REGIONS = np.r_[2200000:2400000,4200000:4400000]
# regions that should be taken from validation
_MODIFIED_VALID_FROM_VALID_REGIONS = np.r_[4200000:4208000]
# ORDER MATTERS!
_MODIFIED_VALID_REGIONS = np.concatenate([_MODIFIED_VALID_FROM_TRAIN_REGIONS, _MODIFIED_VALID_FROM_VALID_REGIONS])

# 227512 for test (chr8 and chr9) for 227512 total rows
_MODIFIED_TEST_REGIONS  = np.r_[2204000*2:2204000*2 + 227512*2] 



# DNase filepath dictionary
dnase_file_dict = {
  "A549": _ENCODE_DATA_PREFIX + "ENCFF001ARO_sorted_A549.bam",
  "HepG2": _ENCODE_DATA_PREFIX + "ENCFF224FMI_sorted_HepG2.bam",
  "K562": _ENCODE_DATA_PREFIX + "ENCFF271LGJ_sorted_K562.bam",
  'GM12878':_ENCODE_DATA_PREFIX + "ENCFF775ZJX_sorted_GM12878.bam",
  'H1-hESC':_ENCODE_DATA_PREFIX + "ENCFF571SSA_sorted_H1heSC.bam",
  'HeLa-S3':_ENCODE_DATA_PREFIX + "ENCFF783TMX_sorted_HeLaS3.bam",
  'HUVEC':_ENCODE_DATA_PREFIX + "ENCFF757PTA_sorted_HUVEC.bam",
  'GM12891':_ENCODE_DATA_PREFIX + "ENCFF070BAN_sorted_GM12891.bam",
  'MCF-7':_ENCODE_DATA_PREFIX + "ENCFF432OZA_sorted_MCF7.bam",
  'GM12892': _ENCODE_DATA_PREFIX + "ENCFF260LKE_sorted_GM12892.bam",
  'HCT-116': _ENCODE_DATA_PREFIX + "ENCFF291HHS_sorted_HCT116.bam"
}
