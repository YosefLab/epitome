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
and the assays you want to use to compute cell type similarity.

.. code:: python

 	targets = ['CTCF','RAD21','SMC3']
	celltypes = ['K562', 'A549', 'GM12878']

	dataset = EpitomeDataset(targets=targets, cells=celltypes)

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
		min_targets_per_cell = 3) # requires that each cell type has data for all three ChIP-seq targets


Note that by default, EpitomeDataset sets DNase-seq (DNase) to be used to compute
cell type similarity between cell types. To specify a different assay to compute
cell type similarity, you can specify in the Epitome dataset:

.. code:: python

	dataset = EpitomeDataset(targets=targets, cells=celltypes, similarity_targets = ['DNase', 'H3K27ac'])

You can then visualize the ChIP-seq targets and cell types in your dataset by
using the ``view()`` function:

.. code:: python

	dataset.view()


To list all of the ChIP-seq targets that an Epitome dataset has available data for,
you can define an Epitome Dataset without specifying ``targets`` or ``cells``.
You can then use the ``list_targets()`` function to print all available ChIP-seq targets
in the dataset:

.. code:: python

	dataset = EpitomeDataset()

	dataset.list_targets() # prints > 200 ChIP-seq targets



You can now use your ``dataset`` in an Epitome model.
