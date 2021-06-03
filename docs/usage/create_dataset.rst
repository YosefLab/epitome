Configuring data
================

Epitome pre-processes ChIP-seq peaks and DNase-seq peaks from ENCODE and ChIP-Atlas for usage
in the Epitome models. Pre-processed datasets for hg19 are lazily downloaded from
`Amazon S3 <https://epitome-data.s3-us-west-1.amazonaws.com/hg19/data.zip>`__
when users run an Epitome model.


Each downloaded dataset contains an h5 file (data.h5). This h5 file contains the following
keys:

- ``data``: a numerical matrix where rows indicate different assays and columns indicate genomic locations

- ``rows``: row information for the data matrix.

  - ``rows/celltypes``: which cell type corresponds to each row

  - ``rows/targets``: which ChIP-seq target corresponds to each row. Can also be DNase-seq

- ``columns``: contains information on the genomic locations that correspond to each

  - ``columns/binSize``: size of genome regions (default is 200bp)

  - ``columns/index/test_chrs``: test chromosomes (default is chrs 8/9)

  - ``columns/index/valid_chrs``: validation chromosomes (default is chr 7)

  - ``columns/index/TEST``: indices that specify the test set

  - ``columns/index/VALID``: indices that specify the validation set

  - ``columns/index/TRAIN``: indices that specify the train set (all autosomal chromosomes, excluding VALID and TEST

  - ``columns/start``: start of each genomic location for each column

  - ``columns/chr``: chromosome for each column

- ``/meta``: metadata for how this dataset was generated

  - ``meta/assembly``: genome assembly

  - ``meta/source``: source for data. Either 'ChIP-Atlas' or 'ENCODE'

Generating data for Epitome from ENCODE
---------------------------------------

You can generate your own Epitome dataset from ENCODE using the following command:
``download_encode.py``.

.. code:: bash

  >> python download_encode.py -h

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
    --bigBedToBed BIGBEDTOBED
                          Path to bigBedToBed executable, downloaded from
                          http://hgdownload.cse.ucsc.edu/admin/exe/


To use your own dataset in an Epitome model, make sure to specify the ``data_dir``
and/or ``assembly`` variables when creating the ``EpitomeDataset`` class. This
will tell Epitome where to load data from. If neither variables are specified,
the default assembly will be downloaded from the Epitome AWS S3 cluster into the
default data directory on your machine. See `Load your processed dataset <./dataset.html>`__ for more details.

.. code:: python

  from epitome.dataset import *

  dataset = EpitomeDataset(data_dir=path/to/configured/data, assembly="hg19")
  ...


Generating data for Epitome from ChIP-Atlas
-------------------------------------------

.. code:: bash

  >> python download_chip_atlas.py -h

  usage: download_chip_atlas.py [-h] [--metadata_url METADATA_URL]
                          [--min_chip_per_cell MIN_CHIP_PER_CELL]
                          [--min_cells_per_chip MIN_CELLS_PER_CHIP]
                          download_path
                          {ce10,ce11,dm3,dm6,hg19,hg38,mm10,mm9,rn6,sacCer3}
                          output_path

  Downloads ChIP-Atlas data from a chip_atlas_experiment_list.csv file.

  positional arguments:
  download_path         Temporary path to download bed/bigbed files to.
  {ce10,ce11,dm3,dm6,hg19,hg38,mm10,mm9,rn6,sacCer3}
                        assembly to filter files in metadata.tsv file by.
  output_path           path to save file data to

  optional arguments:
  -h, --help            show this help message and exit
  --metadata_url METADATA_URL
                        ChIP-Atlas metadata URL.
  --min_chip_per_cell MIN_CHIP_PER_CELL
                        Minimum ChIP-seq experiments for each cell type.
  --min_cells_per_chip MIN_CELLS_PER_CHIP
                        Minimum cells a given ChIP-seq target must be observed
                        in.


TODO: need to add this script as a binary in the module.
