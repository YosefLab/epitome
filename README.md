# Epitome

Full pipeline for learning TFBS from epigenetic datasets.

## Requirements:
-- conda 
-- python 3.3.6

## Setup:
1. Create a conda venv:
/data/akmorrow/anaconda2/bin/conda create --name EpitomeEnv python=3.6

2. setup: pip install -e .

## Making Records
TODO

## Training a Model

```python
from epitome.models import *
radii = [1,3,10,30]
model = MLP(4, [100, 100, 100, 50], 
            tf.tanh, 
            train_data, 
            valid_data, 
            test_data, 
            test_celltypes,
            gen_from_peaks, 
            matrix,
            assaymap,
            cellmap,
            shuffle_size=2, 
            batch_size=64,
            radii=radii)
model.train(10000)
```


