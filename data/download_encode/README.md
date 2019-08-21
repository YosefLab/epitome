# Creating a dataset
 
An Epitome dataset consists of a large n by k matrix, where n is the number of celltype/assay combinations and k is the number of 200bp regions in the genome.

You can create an Epitome dataset for the hg19, GRCh38 or mm10 genome. To create a dataset, run:

```
python data/download_encode/download_encode.py bed_download_path hg19 path_to_bigBedToBed output_dir

```

where path_to_bigBedToBed is a path to the UCSC bigBedToBed executable, which can be downloaded from 
http://hgdownload.cse.ucsc.edu/admin/exe/.


