# Epitome

Pipeline for predicting ChIP-seq peaks in novel cell types using chromatin accessibility.

![Epitome Diagram](https://github.com/YosefLab/epitome/raw/master/docs/figures/epitome_diagram_celllines.png)

Epitome leverages chromatin accessibility (either DNase-seq or ATAC-seq) to predict epigenetic events in a novel cell type of interest. Such epigenetic events include transcription factor binding sites and histone modifications. Epitome computes chromatin accessibility similarity between ENCODE cell types and the novel cell type, and uses this information to transfer known epigentic signal to the novel cell type of interest.


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

TODO: link to documentation

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
