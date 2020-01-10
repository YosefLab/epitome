Training an Epitome Model
=========================

Once you have `installed Epitome and configured the data <../installation/source.html>`__, you are ready to train a model.

Training a Model
----------------

First, import Epitome and specified the `path to Epitome data: <./data.html>`__

.. code:: python

	from epitome.constants import *
	from epitome.models import *
	from epitome.generators import *
	from epitome.functions import *
	from epitome.viz import *

	epitome_data_path = <path_to_dataset>

To train a model, you will need to first specify the assays and cell lines you would like to train with:

.. code:: bash

	matrix, cellmap, assaymap = get_assays_from_feature_file(feature_path=os.path.join(epitome_data_path,'feature_name'),
                                      eligible_assays = None,
                                      eligible_cells = None,
                                      min_assays_per_cell=2,
                                      min_cells_per_assay=2)



Next train a model for 5000 iterations:

.. code:: python

  	# for each, train a model
	shuffle_size = 2

	model = MLP(epitome_data_path,
	            ["K562"], # cell line reserved for testing
	            matrix,
	            assaymap,
	            cellmap,
	            shuffle_size=shuffle_size,
	            prefetch_size = 64,
	            debug = False,
	            batch_size=64)

	model.train(5000)

You can then evaluate model performance on held out test cell lines specified in the model declaration. In this case, we will evaluate on K562 on the first 10,000 points.


.. code:: python

  results = model.test(self, 10000, log=True)

The output of `results` will contain the predictions and truth values, a dictionary of assay specific performance metrics, and the average auROC and auPRC across all evaluated assays.
