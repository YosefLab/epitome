Training an Epitome Model
=========================

Once you have `installed Epitome <../installation/source.html>`__, you are ready to train a model.

Training a Model
----------------

First, import Epitome:

.. code:: python

	from epitome.dataset import *
	from epitome.models import *

Create an Epitome Dataset
-------------------------

First, create an Epitome Dataset. In the dataset, you will define the
ChIP-seq targets you want to predict, the cell types you want to train from,
and the assays you want to use to compute cell type similarity. For more information
on creating an Epitome dataset, see `Configuring data <./dataset.html>`__.

.. code:: python

 	targets = ['CTCF','RAD21','SMC3']
	celltypes = ['K562', 'A549', 'GM12878']

	dataset = EpitomeDataset(targets=targets, cells=celltypes)

Now, you can create a model:

.. code:: python

	model = EpitomeModel(dataset, test_celltypes = ["K562"]) # cell line reserved for testing

Next, train the model. Here, we train the model for 5000 iterations:

.. code:: python

	model.train(5000)

You can then evaluate model performance on held out test cell lines specified in the model declaration. In this case, we will evaluate on K562 on the first 10,000 points.

.. code:: python

	results = model.test(10000,
		mode = Dataset.TEST,
		calculate_metrics=True)

The output of `results` will contain the predictions and truth values, a dictionary of assay specific performance metrics, and the average auROC and auPRC across all evaluated assays.
