# Copies over required motifs from s124 to shared NFS.
# These files are being copied from /data/yosef2/GenomicRegionsDB/hg19/motifs/ on s124.
# This is required to run DAStk metrics, but should only be run once.

# Imports
from shutil import copyfile
from epitome.functions import *

motif_db_src = "/data/yosef2/GenomicRegionsDB/hg19/motifs/"
motif_db_dest = "/home/eecs/akmorrow/epitome/motif_db"

matrix, cellmap, assaymap = get_assays_from_feature_file(feature_path='../../data/feature_name',eligible_assays = None,
                                  eligible_cells = None, min_cells_per_assay = 2, min_assays_per_cell=2)

factors = list(map(lambda x: x.upper(), list(assaymap)))[1:]

list(map(lambda x: x.split("_")[0], os.listdir(motif_db_src)))

# select motif files that are in the list of factors and copy them over from s124
files = list(filter(lambda x: x.split("_")[0] in factors, os.listdir(motif_db_src)))

if (len(files) == 0):
    print("No motif files for these factors")

for file in files:  
    copyfile(os.path.join(motif_db_src, file), os.path.join(motif_db_dest, file))