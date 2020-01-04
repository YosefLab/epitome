# Epitome

Full pipeline for learning TFBS from epigenetic datasets.

![Epitome Diagram](docs/figures/epitome_diagram.png)

Epitome leverages chromatin accessibility data to predict transcription factor binding sites on a novel cell type of interest. Epitome computes the chromatin similarity between 11 cell types in ENCODE and the novel cell types, and uses chromatin similarity to transfer binding information in known cell types to a novel cell type of interest. 


## Requirements:
* [conda](https://docs.conda.io/en/latest/miniconda.html)
* python > 3.6

## Setup:
1. Create and activate a conda venv:
```
conda create --name EpitomeEnv python=3.6
source activate EpitomeEnv
```
2. setup: 
```
pip install -e .
```

Note: Epitome is configured for tensorflow 1.12/Cuda 9. If you have a different
version of cuda, update tensorflow-gpu version accordingly.

To check your Cuda version:
```
nvcc --version
```

## Configuring data
To download and format training data, run the bin/get_deepsea_data.py script:


```
python bin/get_deepsea_data.py 
usage: get_deepsea_data.py [-h] --output_path OUTPUT_PATH
```

## Training a Model

```python

    from epitome.models import *
    model = VLP(data
            test_celltypes,
            matrix,
            assaymap,
            cellmap,
            shuffle_size=2, 
            batch_size=64)
    model.train(10000)
```


