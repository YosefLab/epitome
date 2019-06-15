.. epitome documentation master file, created by

Introduction
============

Epitome is a computation model that leverages chromatin accessibility data to predict transcription factor binding sites on a novel cell type of interest. Epitome computes the chromatin similarity between 11 cell types in ENCODE and the novel cell types, and uses chromatin similarity to transfer binding information in known cell types to a novel cell type of interest.

.. image:: figures/epitome_diagram.png


.. toctree::
   :caption: Installation
   :maxdepth: 2

   installation/source

.. toctree::
   :caption: Usage and Examples
   :maxdepth: 2

   usage/train
   usage/predict



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
