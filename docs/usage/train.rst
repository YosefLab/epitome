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

	dataset = EpitomeDataset(targets=targets, cells=celltypes, data_dir=path/to/configured/data)

Train a Model
----------------
Now, you can create a model:

.. code:: python

	model = EpitomeModel(dataset, test_celltypes = ["K562"]) # cell line reserved for testing

Next, train the model. Here, we train the model for 5000 batches:

.. code:: python

	model.train(5000)

Train a Model that Stops Early
-------------------------------
If you are not sure how many batches your model should train for or are concerned
about your model overfitting, you can specify the max_valid_batches parameter when
initializing the model, which will create a train_validation dataset the size of
max_valid_batches. This forces the model to validate on the train-validation dataset
and generate the train-validation loss every 200 training batches. The model may
stop training early (before max_train_batches) if the model's train-validation
losses stop improving during training. Else, the model will continue to train
until max_train_batches.

First, we have created a model that has a train-validation set size of 1000:

.. code:: python

	model = EpitomeModel(dataset,
		test_celltypes = ["K562"], # cell line reserved for testing
		max_valid_batches = 1000) # train_validation set size reserved while training

Next, we train the model for a maximum of 5000 batches. If the train-validation
loss stops improving, the model will stop training early:

.. code:: python

	best_model_batches, total_trained_batches, train_valid_losses = model.train(5000)

If you are concerned about the model above overtraining because the model continues
to improve by miniscule amounts, you can specify the min-delta which is minimum
change in the train-validation loss required to qualify as an improvement. In the
model below, a minimum improvement of at least 0.01 is required for the model to
qualify as improving.

If you are concerned about the model above under-fitting (stopping training too
early because the train-validation loss might worsen slightly before reaching it's
highest accuracy), you can specify the patience. In the model below, specifying
a patience of 3 allows the model to train for up to 3 train-validation iterations
(200 batches each) with no improvement, before stopping training.

You can read the in-depth explanation of these hyper-parameters in
`this section <https://www.overleaf.com/project/5cd315cb8028bd409596bdff>`__ of the
paper. Detailed documentation of the train() function can also
be found in the `Github repo <https://github.com/YosefLab/epitome>`__.

.. code:: python

	best_model_batches, total_trained_batches, train_valid_losses = model.train(5000,
		patience = 3,
		min_delta = 0.01)

Test the Model
----------------
Finally, you can evaluate model performance on held out test cell lines specified
in the model declaration. In this case, we will evaluate on K562 on the first 10,000 points.

.. code:: python

	results = model.test(10000,
		mode = Dataset.TEST,
		calculate_metrics=True)

The output of `results` will contain the predictions and truth values, a dictionary
of assay specific performance metrics, and the average auROC and auPRC across all
evaluated assays.
