Installing Epitome
==================

**Note**: Epitome is configured for tensorflow 2/Cuda 9. If you have a different
version of cuda, update tensorflow-gpu version accordingly.

Requirements
------------

* `conda <https://docs.conda.io/en/latest/miniconda.html>`__
* python 3.7
* `tensorflow 2.3.0 <https://www.tensorflow.org/install/source>`__


Installing Tensorflow
---------------------

In order to run Epitome as efficiently as possible, you should install
`tensorflow from source <https://www.tensorflow.org/install/source>`__.
If you have not installed tensorflow prior to installing Epitome, Epitome will
install tensorflow from pip, which will not be optimized for your hardware.


Installation from Pip
---------------------

1. Create and activate a pytion 3.7 conda venv:

.. code:: bash

	conda create --name EpitomeEnv python=3.7
	source activate EpitomeEnv

2. Install Epitome from Pypi:

.. code:: bash

	pip install epitome

Installation from Source
------------------------

1. Create and activate a pytion 3.7 conda venv:

.. code:: bash

	conda create --name EpitomeEnv python=3.7
	source activate EpitomeEnv


2. Get Epitome code:

.. code:: bash

	git clone https://github.com/YosefLab/epitome.git
	cd epitome


3. Install Epitome and its requirements

.. code:: bash

	pip install -e .
