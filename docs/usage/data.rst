Configuring data
================

Epitome pre-processes ChIP-seq peaks and DNase-seq peaks from ENCODE for usage
in the Epitome models. Pre-processed datasets are lazily downloaded from `Amazon S3 <../https://epitome-data.s3-us-west-1.amazonaws.com/data.zip>`__ when users run an Epitome model.


This dataset contains the following files:

- **train.npz, valid.npz, and test.npz**: compressed numpy data matrices. Valid.npz includes chr7 data, test.npz includes chr8 and chr9, and train.npz includes data from all other chromosomes.

- **all.pos.bed.gz**: gunzipped genomic regions matching the numpy data matrices.

- **feature_name**: ChIP-seq and DNase-seq peaks corresponding to the data matrix.


Generating data for Epitome
---------------------------

You can generate your own Epitome dataset from ENCODE using the following command:
```download_encode.py```.

.. code:: bash

  python download_encode.py -h

  usage: download_encode.py [-h] [--metadata_url METADATA_URL]
                          [--min_chip_per_cell MIN_CHIP_PER_CELL]
                          [--regions_file REGIONS_FILE]
                          download_path {hg19,mm10,GRCh38} bigBedToBed
                          output_path

TODO: need to add this script as a binary in the module.
