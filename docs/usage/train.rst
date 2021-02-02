Training an Epitome Model
=========================

Once you have `installed Epitome <../installation/source.html>`__, you are ready to train a model.

Create a Dataset
----------------

First, import Epitome:

.. code:: python

	from epitome.dataset import *
	from epitome.models import *

Next, create an Epitome Dataset. In the dataset, you will need to define the
ChIP-seq targets you want to predict, the cell types you want to train from,
and the assays you want to use to compute cell type similarity. For more information
on creating an Epitome dataset, see `Configuring data <./dataset.html>`__.

.. code:: python

 	targets = ['CTCF','RAD21','SMC3']
	celltypes = ['K562', 'A549', 'GM12878']

	dataset = EpitomeDataset(targets, celltypes)

Train a Model
----------------
Now, you can create a model:

.. code:: python

	model = VLP(dataset, test_celltypes = ["K562"]) # cell line reserved for testing

Next, train the model. Here, we train the model for 5000 iterations:

.. code:: python

	model.train(5000)

Train a Model that Stops Early
----------------
If you are not sure how many iterations your model should train for, you can allow the model to train until either the train-validation losses converge or the maximum train iterations (num_steps) are reached-- whichever comes first.

First, you can create a model and specify the number of max_valid_batches reserved while training. During training, the model computes the loss on the max_valid_batches set every 200 iterations, and its loss is compared to previous batches. Here, we have reserved 1000 as the max_valid_batches size:

.. code:: python

	model = VLP(dataset,
		test_celltypes = ["K562"], # cell line reserved for testing
		max_valid_batches = 1000) # train_validation set size reserved while training

Next, train the model. Here, we train the model for 5000 iterations with the default patience and min_delta parameters:

.. code:: python

	best_model_steps, num_steps, train_valid_losses = model.train(5000)

If you are concerned about the train-validation loss converging prematurely, you can specify the patience and min_delta parameters:

.. code:: python

	best_model_steps, num_steps, train_valid_losses = model.train(5000,
		patience = 3,
		min_delta = 0.1)

Test the Model
----------------
Finally, you can evaluate model performance on held out test cell lines specified in the model declaration. In this case, we will evaluate on K562 on the first 10,000 points.

.. code:: python

	results = model.test(10000,
		mode = Dataset.TEST,
		calculate_metrics=True)

The output of `results` will contain the predictions and truth values, a dictionary of assay specific performance metrics, and the average auROC and auPRC across all evaluated assays.
