# Epitome

Full pipeline for learning TFBS from epigenetic datasets.

![Epitome Diagram](figures/epitome_diagram.png)

Epitome leverages chromatin accessibility data to predict transcription factor binding sites on a novel cell type of interest. Epitome computes the chromatin similarity between 11 cell types in ENCODE and the novel cell types, and uses chromatin similarity to transfer binding information in known cell types to a novel cell type of interest. 


## Requirements:
-- conda
-- python 3.3.6

## Setup:
1. Create a conda venv:
/data/akmorrow/anaconda2/bin/conda create --name EpitomeEnv python=3.6

2. conda install --yes --file dnase/conda-requirements.txt
3. pip install -r dnase/pip-requirements.txt
