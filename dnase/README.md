 

## Setup

### Install requirements:


#### Git Lfs (for kipoi)

conda install -c conda-forge git-lfs && git lfs install

#### Pip and conda requirements
pip3 install -r pip-requirements.txt
conda install --yes --file conda_requirements.txt


## Running

1. dnase/model_main.ipynb: Train and save model
2. dnase/predict_epitome.py: runs predictions given a set of bed files and reduces down to single binding statistic for each factor.
3. dnase/Metrics/Run-Epitome.py: evaluating model on test set for all transcription factors, using ROC/PR for metrics 
