
# coding: utf-8

# ## Download DNase from ENCODE
# 
# This script uses files.txt and ENCODE metadata to download DNAse for hg38 for specific cell types.
# Because ENCODE does not have hg19 data for ATAC-seq, we have to re-align it from scratch.

# In[69]:

import pandas as pd
import numpy as np
import os
import urllib


# Download for cell types (only A549 has data)
celltypes = ['A549']

files = pd.read_csv("./metadata.tsv", sep="\t")


for celltype in celltypes:

    
    # filter files that have cell type
    files_celltype = files[(files["Assay"] == "ATAC-seq") &
                       (files["Biosample term name"] == celltype) & 
                      (files["File format"] == "fastq") & 
                      (files["Biosample treatments"].isnull())]
    
    print("found %i files for cell type %s " % (files_celltype.shape[0], celltype))
    
    # mkdir for this celltype
    if not os.path.exists(celltype):
        os.mkdir(celltype)
 
    # download to celltype file
    for index, f in files_celltype.iterrows():
        path = f["File download URL"]
        id = f["File accession"]
        outname = "/data/yosef2/Epitome/%s/%s.fastq.gz" % (celltype, id)
        print(outname)
        urllib.urlretrieve(path, filename=outname)
    