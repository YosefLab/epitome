Building Epitome from Source
============================

**Note**: Epitome is configured for tensorflow 2/Cuda 9. If you have a different
version of cuda, update tensorflow-gpu version accordingly.

Requirements
------------

* `conda <https://docs.conda.io/en/latest/miniconda.html>`__
* python 3.7

Installation
------------

1. Create and activate a pytion 3.7 conda venv:

.. code:: bash

	conda create --name EpitomeEnv python=3.7
	source activate EpitomeEnv


2. Get Epitome code:

.. code:: bash

	git clone git@github.com:akmorrow13/epitome.git
	cd epitome


3. Install Epitome and its requirements

.. code:: bash

	pip install -e .



Configuring Data
----------------

Epitome requires data for training, validation and test. See `Configuring Epitome data <../usage/data.html>`__ for more information
on how to download data for Epitome or generate your own dataset.
