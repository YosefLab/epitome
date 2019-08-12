# Defcom

For more information about Defcom, please see their [documentation](https://defcom.readthedocs.io/en/latest/).

## Requirements:

* [conda](https://docs.conda.io/en/latest/miniconda.html)
* python 2.7
* [tabix](http://www.htslib.org/doc/tabix.html)

## Configuring Data

To download the data for DeFCoM, run the evaluation/defcom/get_defcom_data.py script:
The data for DeFCoM consists of a series of files containing active and inactive motif
sites for a variety of cell types and transcription factors. This script will download
these files and then split them up into training, validation, and testing sets so they
can be easily used with both defcom and epitome. Before doing this you should ensure you
have tabix and bgzip installed on your machine.

```
python evaluation/defcom/get_defcom_data.py
usage: get_defcom_data.py [-h] --output_path OUTPUT_PATH
```

## Setup:
1. Create and activate a conda venv:
```
conda create --name DefcomEnv python=2.7
source activate DefcomEnv
```
2. setup: 

You will likely have to jump through some hoops to get Defcom running.
Here's likely what you will have to do to get defcom to run.

```
# install requirements
pip install -r requirements.txt

# you then might have to actually make the files executable in which case you should run
# if you don't know where defcom's train.py is located you can run pip uninstall defcom and
# it should show the location of train.py and predict.py

chmod +x <path_to_defcom/bin/train.py> 
chmod +x <path_to_defcom/bin/predict.py>

# then if your machine doesn't recognize train.py then you will have to add them to your path with something like

PATH=$PATH:<path_to_defcom_bin>
```

Now in order to test whether defcom is able to run on the data that is now on your system
You need DNAse data to feed to defcom as well.

You can find DNAse data [here](http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeOpenChromDnase/)

## Training and Predicting a DefCOM Model

```python
# Here is an example of how to use Defcom and evaluate its results

from evaluation.defcom.defcom_functions import *

train_cell_type = 'k562'
query_cell_type = 'hepg2'
tf = 'rfx5'
active_sites_file    = 'data/defcom_data/rfx5_k562_pos_train.bed'
inactive_sites_file  = 'data/defcom_data/rfx5_k562_neg_train.bed'
candidate_sites_file = 'data/defcom_data/rfx5_hepg2_all_valid.bed'
training_bam_file    = 'data/dnase_data/wgEncodeOpenChromDnaseK562AlnRep1.bam'
candidate_bam_file   = 'data/dnase_data/wgEncodeOpenChromDnaseHepg2AlnRep3.bam'
model_out            = 'data/defcom_models'
results_out          = 'data/defcom_results'
config_out           = 'data/defcom_configs'


config_file, prediction_results_file = createConfigFile(train_cell_type,
                                                tf,
                                                query_cell_type, 
                                                active_sites_file, 
                                                inactive_sites_file, 
                                                candidate_sites_file, 
                                                training_bam_file, 
                                                candidate_bam_file,
                                                model_out,
                                                results_out,
                                                config_out)

print('Training')
train_defcom(config_file)
print('Predicting')
predict_defcom(config_file)
print('Done')

data_type = 'valid' # the dataset we want to evaluate on, in this case its validation
defcom_data_dir = 'data/defcom_data' # the local data directory specfied when you downloaded the defcom data

auROC, auPRC = evaluateDefcomModel(query_cell_type, tf, data_type, prediction_results_file, defcom_data_dir)
```

## Visualizing the Results

### Creating Scatterplots

```python
import pandas as pd
from evaluation.defcom.visualization import *

# to start: load in the data, should be in the format of 
# model | training_cell | transcription_factor | query_cell | auROC | auPR

defcom_eval_results = 'results/eval_results/defcom_functions_test_results.csv'

results = pd.read_csv(defcom_eval_results, index_col=False)
# let's say for example we want to exlcude results iwth Ep300 and Gabpa because we only have results for them for DeFCoM
results = results.loc[ (results['transcription_factor'] != 'Ep300') & (results['transcription_factor'] != 'Gabpa')]

generateScatterPlots(results, 'auROC')

```

### Creating Boxplots

```python
import pandas as pd
from evaluation.defcom.visualization import *

# to start: load in the data, should be in the format of 
# model | training_cell | transcription_factor | query_cell | auROC | auPR

# to start: let's go ahead and load in the data, this could be the same predictions results file as the above example
defcom_eval_results = 'results/eval_results/defcom_functions_test_results.csv'

results = pd.read_csv(defcom_eval_results, index_col=False)
# let's say for example we want to exlcude results iwth Ep300 and Gabpa because we only have results for them for DeFCoM
results = results.loc[ (results['transcription_factor'] != 'Ep300') & (results['transcription_factor'] != 'Gabpa')]

generateComparativeBoxplot(results, 'auROC', 'results/plots/defcom_functions_test/comparative_boxplot_auROC.png')
generateComparativeBoxplot(results, 'auPR', 'results/plots/defcom_functions_test/comparative_boxplot_auPR.png')
```



## Some things to note

The training_bam_file and candidate_bam_file must be BAM files for DNase-seq/ATAC-seq read alignments to be used for model training. The file must have an index file with the same name but with the ‘.bai’ extension appended.

The training sites and candidate sites must be in pseudo-BED format and in the same directory as this file there must be a gzipped (.gz) file and tabix index (.tbi) file with the same file name prefix.

For more on Defcom's configuration file and inputs see their [documenation on their configuration file](https://defcom.readthedocs.io/en/latest/config/)

Lastly, and very importantly, if using Defcom within a jupyter notebook, keep in mind that the functions that call defcom use the subprocess python module. This means that the environment in which you start python in will be what's used. So if you start a jupyter notebook with a kernel, and then add train.py and predict.py to your PATH variable, then try to train defcom, the subprocess module will be using your OLD PATH variable and will be unable to locate train.py and predict.py.

