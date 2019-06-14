Building Epitome from Source
============================

**Note**: Epitome is configured for tensorflow 1.12/Cuda 9. If you have a different
version of cuda, update tensorflow-gpu version accordingly.

Requirements
------------

* `conda <https://docs.conda.io/en/latest/miniconda.html>`__
* python 3.6

Installation
------------

1. Create and activate a pytion 3.6 conda venv:

.. code:: bash

	conda create --name EpitomeEnv python=3.6
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

To download and format training data, run ```bin/get_deepsea_data.py```:

.. code:: bash

	python bin/get_deepsea_data.py 
	usage: get_deepsea_data.py [-h] --output_path OUTPUT_PATH
