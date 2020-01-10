Configuring data
================

Epitome pre-processes ChIP-seq peaks and DNase-seq peaks from ENCODE for usage
in the Epitome models. Pre-processed data can be downloaded from:

TODO: upload to AWS

This data contains the following:
- train.npz, valid.npz, and test.npz: compressed numpy data matrices. Valid.npz includes chr7 data, test.npz includes chr8 and chr8,
and train.npz includes data from all other chromosomes.
- all.pos.bed.gz: gunzipped genomic regions matching the numpy data matrices
- feature_name: ChIP-seq and DNase-seq peaks corresponding to the data matrix.


Generating data for Epitome
---------------------------

TODO: need to add this script as a binary in the module.

You can generate your own Epitome dataset from ENCODE using the following command:
```download_encode.py```.

.. code:: bash

  python get_deepsea_data.py -h

  usage: download_encode.py [-h] [--metadata_url METADATA_URL]
                          [--min_chip_per_cell MIN_CHIP_PER_CELL]
                          [--regions_file REGIONS_FILE]
                          download_path {hg19,mm10,GRCh38} bigBedToBed
                          output_path
