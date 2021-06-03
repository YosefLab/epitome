Creating an Epitome Dataset
===========================

This section explains how to load in an Epitome Dataset. If you
are interested in pre-processing your own dataset from ENCODE or
ChIP-Atlas, see `Configuring data <./create_dataset.html>`__.

First, import EpitomeDataset:

.. code:: python

	from epitome.dataset import *

Create an Epitome Dataset
-------------------------

First, create an Epitome Dataset. In the dataset, you will define the
ChIP-seq targets you want to predict, the cell types you want to train from,
the assays you want to use to compute cell type similarity, and the data directory
to where you configured your pre-processed data in ``path/to/configured/data``.

.. code:: python

 	targets = ['CTCF','RAD21','SMC3']
	celltypes = ['K562', 'A549', 'GM12878']

	dataset = EpitomeDataset(targets=targets, cells=celltypes, data_dir=path/to/configured/data)

Note that you do not have to define ``celltypes``. If you leave ``celltypes``
blank, the Epitome dataset will choose cell types that have coverage  for the
ChIP-seq targets chosen. The parameters ``min_cells_per_target`` and ``min_targets_per_cell``
specify the minimum number of cells required for a ChIP-seq target, and the minimum
number of ChIP-seq targets required to include a celltype. By default,
``min_cells_per_target = 3`` and ``min_targets_per_cell = 2``.

.. code:: python

 	targets = ['CTCF','RAD21','SMC3']

	dataset = EpitomeDataset(targets=targets,
		min_cells_per_target = 4, # requires that each ChIP-seq target has data from at least 4 cell types
		min_targets_per_cell = 3, # requires that each cell type has data for all three ChIP-seq targets
		data_dir = path/to/configured/data)


Note that by default, EpitomeDataset sets DNase-seq (DNase) to be used to compute
cell type similarity between cell types. To specify a different assay to compute
cell type similarity, you can specify in the Epitome dataset:

.. code:: python

	dataset = EpitomeDataset(targets=targets,
					cells=celltypes,
					similarity_targets = ['DNase', 'H3K27ac'],
					data_dir=path/to/configured/data)

You can then visualize the ChIP-seq targets and cell types in your dataset by
using the ``view()`` function:

.. code:: python

	dataset.view()


To list all of the ChIP-seq targets that an Epitome dataset has available data for,
you can define an Epitome Dataset without specifying ``targets`` or ``cells``.
You can then use the ``list_targets()`` function to print all available ChIP-seq targets
in the dataset:

.. code:: python

	dataset = EpitomeDataset(data_dir=path/to/configured/data)

	dataset.list_targets() # prints > 200 ChIP-seq targets



You can now use your ``dataset`` in an Epitome model.

Load your processed dataset
-------------------------
You can specify the data path and/or genome assembly that you would like to use
in the Epitome dataset. You just need to define the ``data_dir`` and/or
``assembly`` variables:

.. code:: python

	dataset = EpitomeDataset(data_dir="~/$USERNAME/epitome/data",
				assembly="hg19")

Note if both the ``data_dir`` and ``assembly`` are set, the dataset will
append the specified assembly to the data_dir path such as
``~/$USERNAME/epitome/data/hg19/data.h5`` and return the dataset that is stored
in the path if it exists. If there is no data stored at that path, Epitome will
try to download the specified assembly from the S3 cluster at
https://epitome-data.s3-us-west-1.amazonaws.com.

You do not need to define both variables though. If you leave ``data_dir`` empty,
the Epitome dataset will append the ``assembly`` to the default data path located
in ``~/$USER_NAME/.epitome/data/`` and return the dataset if it exists at that path.
If there is no existing dataset located at the data path, Epitome will download
the dataset for the specified assembly from S3 to that path:

.. code:: python

	dataset = EpitomeDataset(assembly="hg19")

If the assembly is not specified but the ``data_dir`` is, the dataset will assume
that the specified data directory ``data_dir`` is the absolute data path and it
will append the default assembly to the configured data path. Like above, if the
dataset exists at the configured data path, Epitome will load the configured data
into the EpitomeDataset. If there is no existing dataset, Epitome will download
the dataset for the default assembly from S3 and store it at the default data path:

.. code:: python

	dataset = EpitomeDataset(data_dir=path/to/configured/data)

If neither ``data_dir`` or ``assembly`` are set, the dataset will just try to
fetch the ``data.zip`` file in the default data directory. If no data exists in
the default directory, Epitome will download the dataset for the default assembly
from S3 and store it at the default data path:

.. code:: python

	dataset = EpitomeDataset()
