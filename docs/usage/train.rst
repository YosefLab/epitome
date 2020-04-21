Training an Epitome Model
=========================

Once you have `installed Epitome <../installation/source.html>`__, you are ready to train a model.

Training a Model
----------------

First, import Epitome:

.. code:: python

	from epitome.constants import *
	from epitome.models import *
	from epitome.generators import *
	from epitome.functions import *
	from epitome.viz import *

Quick Start
^^^^^^^^^^^

First, define the assays you would like to train. Then you can create a `VLP` model:

.. code:: python

	print(list_assays()) # prints all targets that epitome can train

	assays = ['CTCF','RAD21','SMC3']
	model = VLP(assays, test_celltypes = ["K562"]) # cell line reserved for testing

To train a model on a specific set of targets and cell lines, you will need to first specify the assays and cell lines you would like to train with:

.. code:: bash

	matrix, cellmap, assaymap = get_assays_from_feature_file(eligible_assays = None,
		eligible_cells = None,
		min_assays_per_cell=6,
		min_cells_per_assay=8)

	# visualize cell lines and ChIP-seq peaks you have selected
	plot_assay_heatmap(matrix, cellmap, assaymap)


Next define a model:

.. code:: python

	model = VLP(list(assaymap),
		matrix = matrix,
		assaymap = assaymap,
		cellmap = cellmap,
		test_celltypes = ["K562"]) # cell line reserved for testing)


Next, train the model. Here, we train the model for 5000 iterations:

.. code:: python

	model.train(5000)

You can then evaluate model performance on held out test cell lines specified in the model declaration. In this case, we will evaluate on K562 on the first 10,000 points.


.. code:: python

	results = model.test(10000,
		mode = Dataset.TEST,
		calculate_metrics=True)

The output of `results` will contain the predictions and truth values, a dictionary of assay specific performance metrics, and the average auROC and auPRC across all evaluated assays.
