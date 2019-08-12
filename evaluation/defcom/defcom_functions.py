import subprocess
import pandas as pd
import sklearn.metrics
import os

from matplotlib import pyplot as plt
from matplotlib import lines as mlines
import seaborn as sns

################### Functions to call Defcom ########################
#relative_dir = os.path.dirname(os.path.abspath(__file__))


def train_defcom(config_file):
    """
    Trains a DeFCoM model for a given configuration file.
    
    :param config_file: path to configuration file defcom will use to train the model
    
    :return None but outputs a file for the defcom model in a location specified by the config file
    """
    try:
        #path_to_train = os.path.join(relative_dir, '../defcom/bin/train.py')
        response = subprocess.check_output('train.py {}'.format(config_file), shell=True, stderr =subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        
        
def predict_defcom(config_file):
    """
    Predicts transcription factor binding on a query cell type for a given configuration file
    
    :param config_file: path to configuration file defcom will use to run its predictions
    
    :return None but outputs a psuedo-bed file with DeFCoM's predictions in a location specified by the config file
    """
    try:
        #path_to_predict = os.path.join(relative_dir, '../defcom/bin/predict.py')
        response = subprocess.check_output('predict.py {}'.format(config_file), shell=True, stderr =subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        

        
def createConfigFile(train_cell_type, tf, eval_cell_type, 
                     active_sites_file, inactive_sites_file, 
                     candidate_sites_file, training_bam_file, 
                     candidate_bam_file,
                     model_out_dir, results_out_dir, config_out_dir):
    """
    Creates a configuration file for DeFCoM to use for a given set of parameters.
    To learn more about config files, visit https://defcom.readthedocs.io/en/latest/config/
    
    :param train_cell_type: the training cell type
    :param tf: the transcription factor
    :param eval_cell_type: the query cell type
    :param active_sites_file: Path to file containing active genomic regions in the training cell type
    :param inactive_sites_file: Path to file containing inactive genomic regions in the training cell type
    :param candidate_sites_file: Path to file containing genomic regions to predict on in the query cell type
    :param training_bam_file: Path to bam file corresponding to the training cell type
    :param candidate_bam_file: Path to bam file corresponding to the query cell type
    :param model_out_dir: The directory in which the model file will be outputted to
    :param results_out_dir: The directory in which the results file will be outputted to
    :param config_out_dir: The directory in which the config file should be saved in 
    
    :return the path to the generated config file and the path to the results file that will be created
        when defcom runs its predictions
    """

    # we want to open up the template config:
    config_file_name = '{}_{}_{}.cfg'.format(train_cell_type, tf, eval_cell_type)
    config_file_name = os.path.join(config_out_dir, config_file_name)
    config_file = open(config_file_name, 'w')
    config_file.write('[data]\n') # first let's write [data]
    
    # now let's write 
    config_file.write(
    # Active motif sites for training phase
        ('active_sites_file = {} \n' +
        'inactive_sites_file = {} \n' +  
        'candidate_sites_file = {} \n' +
        'training_bam_file = {} \n' +
        'candidate_bam_file = {} \n').format(
            active_sites_file,
            inactive_sites_file,
            candidate_sites_file,
            training_bam_file,
            candidate_bam_file
        )
    )
        
    # let's write the rest of the file, default arguments for defcom
    path_to_config_template = os.path.dirname(os.path.abspath(__file__))
    path_to_config_template = os.path.join(path_to_config_template, 'config_template.cfg')
    with open(path_to_config_template) as f:
        for line in f.readlines():
            config_file.write(line)

    # now let's write the model out and the results out
    model_file_name   = '{}_{}.pkl'.format(train_cell_type, tf)
    results_file_name = '{}_{}_{}_results.bed'.format(train_cell_type, tf, eval_cell_type)
    model_data_file = os.path.join(model_out_dir, model_file_name)
    results_file    = os.path.join(results_out_dir, results_file_name)
    
    config_file.write(
        ('model_data_file = {0}\n' +
        'results_file = {1}').format(model_data_file, results_file)
    )    
    # save it off
    config_file.close()
    
    return config_file_name, results_file



#####################################################################
################# Functions to Evaluate Models ######################
#####################################################################

def evaluateDefcomResults(tf_name, prediction_results_file, pos_file):
    """
    Evalutes the performance of defcom predictions for a given tf, prediction file, and true positive sites
    
    :param tf: The transcription factor being predicted on. Should be in DEFCOM's format (ex: rfx5, instead of RFX5)
    :param prediction_results_file: Path to defcom's predictions file (output of predict.py)
    :param pos_file: The true positive sites, (defcom's data)
    
    :return auROC and auPR for defcom's predictions
    """

    colNames = ['chrom_name', 'chrom_start', 'chrom_end', 'strand', 'etc_1', 'etc_2', 'etc_3', 'score']
    
    # get the prediction results
    predicted_results_df = pd.read_csv(prediction_results_file, sep='\t', names=colNames, index_col=False)
    predicted_results_df.sort_values(colNames[:2], axis = 0, inplace=True)

    # get the true values for active and inactive sites
    pos_sites_df = pd.read_csv(pos_file, sep='\t', names=colNames, index_col=False)
    pos_sites_df = pos_sites_df[['chrom_name', 'chrom_start']]
    pos_sites_df['score'] = 1 # set the values of the pos sites to 1
        
    # merge the results
    result_df = predicted_results_df.merge(pos_sites_df, how='left', left_on=['chrom_name', 'chrom_start'], right_on=['chrom_name', 'chrom_start'])
    result_df = result_df.fillna(0)
    
    # get the predicted values and actual scores
    predicted_scores = result_df['score_x']
    actual_scores = result_df['score_y']
    
    # calculate auROC and auPR
    auROC = sklearn.metrics.roc_auc_score(actual_scores, predicted_scores, average ='macro') # I think pos_label is 1 by default
    auPRC = sklearn.metrics.average_precision_score(actual_scores, predicted_scores, average='macro')
    
    return auROC, auPRC



def evaluateEpitomeResults(tf, prediction_results_file, pos_file):
    """
    Evalutes the performance of epitome predictions for a given tf, prediction file, and true positive sites
    
    :param tf: The transcription factor being predicted on. Should be in epitome's format (ex: RFX5, instead of rfx5)
    :param prediction_results_file: Path to epitome's predictions file (output of score peak file)
    :param pos_file: The true positive sites, (defcom's data)
    
    :return auROC and auPR for epitome's predictions
    """
    # maybe move this
    colNames = ['chrom_name', 'chrom_start', 'chrom_end', 'strand', 'etc_1', 'star', 'score'] # is score supposed to be there

    # get the prediction results data frame
    predicted_results_df = pd.read_csv(prediction_results_file, sep=',', index_col=False)
    predicted_results_df.columns = [str(col_name) for col_name in predicted_results_df.columns]
    predicted_results_df.rename(columns={'0': 'chrom_name', '1': 'chrom_start', '2':'chrom_end'}, inplace=True)

    # get the true values for active and inactive sites
    pos_sites_df = pd.read_csv(pos_file, names=colNames, sep='\t', index_col=False)    
    pos_sites_df = pos_sites_df[['chrom_name','chrom_start']] # keep only columns we need to merge on
    pos_sites_df['score'] = 1 # set the values of the pos sites to 1

    # merge the dataframes
    result_df = predicted_results_df.merge(pos_sites_df, how='left', left_on=['chrom_name','chrom_start'], right_on=['chrom_name','chrom_start'])
    result_df = result_df.fillna(0) # set the inactive sites to 0

    # get the predicted and actual scores
    #predicted_scores = result_df[tf]
    predicted_scores = result_df.iloc[:,2]
    actual_scores = result_df['score']

    # calculate auROC and auPR
    auROC = sklearn.metrics.roc_auc_score(actual_scores, predicted_scores, average ='macro') # I think pos_label is 1 by default
    auPR = sklearn.metrics.average_precision_score(actual_scores, predicted_scores, average='macro')
    
    return auROC, auPR


