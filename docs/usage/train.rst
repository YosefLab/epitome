Training an Epitome Model
=========================

Once you have `installed Epitome and configured the data <../installation/source.rst>`__, you are ready to train a model.

Training a Model
----------------

Load in a configuration file that contains the path to your data and the path to the feature file:

The config file should look something like:

.. code::

	data_path: epitome_data/deepsea_labels_train/
	feature_name_file: eptiome/data/feature_name

Now load in your config file:

.. code:: python

	import yaml

	with open('<path_to_my>/config.yml') as f:
    	config = yaml.safe_load(f)

First, import Epitome and load in data:

.. code:: python

	from epitome.constants import *
	from epitome.models import *
	from epitome.generators import *
	from epitome.functions import *
	from epitome.viz import *

	train_data, valid_data, test_data = load_deepsea_label_data(config["data_path"])
	data = {Dataset.TRAIN: train_data, Dataset.VALID: valid_data, Dataset.TEST: test_data}

To train a model, you will need to first specify the assays and cell lines you would like to train with:


.. code:: bash

	matrix, cellmap, assaymap = get_assays_from_feature_file(feature_path=config['feature_name_file'], 
                                      eligible_assays = None,
                                      eligible_cells = None, 
                                      min_assays_per_cell=2, 
                                      min_cells_per_assay=2)



Next train a model for 5000 iterations:

.. code:: python

  	# for each, train a model
	shuffle_size = 2 

	model = MLP(data,
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

The output of ```results``` will contain the predictions and truth values, a dictionary of assay specific performance metrics, and the average auROC and auPRC across all evaluated assays.



