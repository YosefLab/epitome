Configuring data
================

Epitome pre-processes ChIP-seq peaks and DNase-seq peaks from ENCODE for usage
in the Epitome models. Pre-processed datasets hg19 are lazily downloaded from `Amazon S3 <../https://epitome-data.s3-us-west-1.amazonaws.com/data.zip>`__ when users run an Epitome model.


This dataset contains the following files:

- **train.npz, valid.npz, and test.npz**: compressed numpy data matrices. Valid.npz includes chr7 data, test.npz includes chr8 and chr9, and train.npz includes data from all other chromosomes.

- **all.pos.bed.gz**: gzipped genomic regions matching columns in the numpy data matrices.

- **feature_name**: ChIP-seq and DNase-seq assays corresponding to rows in the data matrices.

The downloaded data can be accessed under :code:`~/.epitome/`.


Generating data for Epitome
---------------------------

You can generate your own Epitome dataset from ENCODE using the following command:
``download_encode.py``.

.. code:: bash

  python download_encode.py -h

    positional arguments:
    download_path         Temporary path to download bed/bigbed files to.
    {hg19,mm10,GRCh38}    assembly to filter files in metadata.tsv file by.
    output_path           path to save file data to

  optional arguments:
    -h, --help            show this help message and exit
    --metadata_url METADATA_URL
                          ENCODE metadata URL.
    --min_chip_per_cell MIN_CHIP_PER_CELL
                          Minimum ChIP-seq experiments for each cell type.
    --min_cells_per_chip MIN_CELLS_PER_CHIP
                          Minimum cells a given ChIP-seq target must be observed
                          in.
    --regions_file REGIONS_FILE
                          File to read regions from
    --bgzip BGZIP         Path to bgzip executable
    --bigBedToBed BIGBEDTOBED
                          Path to bigBedToBed executable, downloaded from
                          http://hgdownload.cse.ucsc.edu/admin/exe/


To use your own dataset in an Epitome model, make sure to set the environment environment variable
``EPITOME_DATA_PATH`` that points to your custom dataset. This will tell Epitome where to load
data from.

.. code:: bash

  import os
  os.environ["EPITOME_DATA_PATH"] = 'path/to/my/epitome/dataset'
  ...

TODO: need to add this script as a binary in the module.
