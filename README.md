[![pypi](https://img.shields.io/pypi/v/epitome.svg)](https://pypi.org/project/epitome/)
[![docs](https://readthedocs.org/projects/epitome/badge/?version=latest)](https://epitome.readthedocs.io/en/latest/)
![Build status](https://github.com/YosefLab/epitome/workflows/epitome/badge.svg)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/6c2cef0a2eae45399c9caed2d8c81965)](https://app.codacy.com/gh/YosefLab/epitome?utm_source=github.com&utm_medium=referral&utm_content=YosefLab/epitome&utm_campaign=Badge_Grade)


# Epitome

Pipeline for predicting ChIP-seq peaks in novel cell types using chromatin accessibility.

![Epitome Diagram](https://github.com/YosefLab/epitome/raw/master/docs/figures/epitome_diagram_celllines.png)

Epitome leverages chromatin accessibility (either DNase-seq or ATAC-seq) to predict epigenetic events in a novel cell type of interest. Such epigenetic events include transcription factor binding sites and histone modifications. Epitome computes chromatin accessibility similarity between ENCODE cell types and the novel cell type, and uses this information to transfer known epigentic signal to the novel cell type of interest.

# Documentation

Epitome documentation is hosted at [readthedocs](https://epitome.readthedocs.io/en/latest/). Documentation for Epitome includes tutorials for creating Epitome datasets, training, testing, and evaluated models.


## Requirements
* [conda](https://docs.conda.io/en/latest/miniconda.html)
* python >= 3.6

## Setup and Installation
1. Create and activate a conda environment:
```
conda create --name EpitomeEnv python=3.6 pip
source activate EpitomeEnv
```
2. Install Epitome:
```
pip install epitome
```


## Training a Model

First, create an Epitome dataset that defines the cell types and ChIP-seq
targets you want to train on,


```python

    from epitome.dataset import *

    targets = ['CTCF','RAD21','SMC3']
    celltypes = ['K562', 'A549', 'GM12878']

    dataset = EpitomeDataset(targets=targets, cells=celltypes)

```

Now, you can create and train your model:

```python

    from epitome.models import *

    model = EpitomeModel(dataset, test_celltypes = ["K562"])
    model.train(5000) # train for 5000 batches
```

## Evaluate a Model:

```python

   model.test(1000) # evaluate how well the model performs on a validation chromosome

```

## Using Epitome on your own dataset:

Epitome can perform genome wide predictions or region specific predictions on
a sample that has either DNase-seq or ATAC-seq.

To score specific regions:

```python

   chromatin_peak_file = ... # path to peak called ATAC-seq or DNase-seq in bed format
   regions_file = ...        # path to bed file of regions to score
   results = model.score_peak_file([chromatin_peak_file], regions_file)

```

To score on the whole genome:

```python

   chromatin_peak_file = ... # path to peak called ATAC-seq or DNase-seq in bed format
   file_prefix = ...        # file to save compressed numpy predictions to.
   model.score_whole_genome([chromatin_peak_file], file_prefix)

```


# Install Epitome for development

To build Epitome for development, run:

```
make develop
```

## Running unit tests

```
make test
```
