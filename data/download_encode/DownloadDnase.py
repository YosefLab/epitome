
# coding: utf-8

# ## Download DNase from ENCODE
# 
# This script uses files.txt and ENCODE metadata to download DNAse for hg19 for specific cell types.

# In[69]:

import pandas as pd
import numpy as np
import os
import urllib


# In[60]:

# Download for cell types
celltypes = ['K562', 'GM12878', 'H1-hESC', 'HepG2', 'HeLa-S3', 'A549',  "endothelial cell of umbilical vein", 'GM12891', 'MCF-7', 'GM12892', 'HCT116']


# In[7]:

files = pd.read_csv("./metadata.tsv", sep="\t")


# In[71]:

for celltype in celltypes:

    
    # filter files that have cell type
    files_celltype = files[(files["Assay"] == "DNase-seq") & 
                           (files["Biosample term name"] == celltype) & 
                           (files["File format"] == "bam") & 
                           (files["Output type"] == "alignments")& 
                           (files["Biosample treatments"].isnull()) &
                            (files["Assembly"] == "hg19") & 
                           (files["Audit ERROR"].isnull()) & 
                            (files["Audit NOT_COMPLIANT"].isnull())]
    
    if (files_celltype.shape[0] == 0):
        files_celltype = files[(files["Assay"] == "DNase-seq") & 
                       (files["Biosample term name"] == celltype) & 
                       (files["File format"] == "bam") & 
                       (files["Output type"] == "alignments")& 
                       (files["Biosample treatments"].isnull()) &
                        (files["Assembly"] == "hg19") & 
                       (files["Audit ERROR"].isnull())]
    
    print("found %i files for cell type %s " % (files_celltype.shape[0], celltype))
    
    # mkdir for this celltype
    if not os.path.exists(celltype):
        os.mkdir(celltype)
 
    # download to celltype file
    for index, f in files_celltype.iterrows():
        path = f["File download URL"]
        id = f["File accession"]
        outname = "%s/%s.bam" % (celltype, id)
        print(outname)
        urllib.urlretrieve(path, filename=outname)
    
    

