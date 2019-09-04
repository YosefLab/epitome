#####################################################################################
## Metric functions for evaluating model.
#####################################################################################

import sklearn.metrics
import numpy as np

# disable sklearn warnings when training
import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def gini(actual, pred, sample_weight):                                                 
    # sort by preds
    df = np.vstack([actual,pred])
    df = df[:,(-pred).argsort()]
    random = ( np.arange(df.shape[1]) + 1 )/ float(df.shape[1])    
    # sum of actual
    totalPos = np.sum(actual)     
    cumPosFound = np.cumsum(df[0])       
    Lorentz = cumPosFound / totalPos
    Gini = Lorentz - random
    Gini[np.where(sample_weight == 0)[0]] = 0
    return np.sum(Gini)    

def gini_normalized(actual, pred, sample_weight = None):              
    normalized_gini = gini(actual, pred, sample_weight)/gini(actual, actual, sample_weight)      
    return normalized_gini       


def get_performance(assaymap, preds, truth, sample_weight):
    
    
    assert(preds.shape == truth.shape)
    assert(preds.shape == sample_weight.shape)
    
    inv_assaymap = {v: k for k, v in assaymap.items()}
    
    evaluated_assays = {}

    for j in range(preds.shape[1]): # for all assays
        # sample_weight mask can only work on 1 row at a time.
        # If a given assay is not available for evaluation, sample_weights will all be 0 
        # and the resulting roc_auc_score will be NaN.
            
        try:
            roc_score = sklearn.metrics.roc_auc_score(truth[:,j], 
                                                      preds[:,j], 
                                                      sample_weight = sample_weight[:, j],
                                                      average='macro')


        except ValueError:
            roc_score = np.NaN
            
        try:
            pr_score = sklearn.metrics.average_precision_score(truth[:,j],
                                                               preds[:,j], 
                                                               sample_weight = sample_weight[:, j])


        except ValueError:
            pr_score = np.NaN

        try:
            gini_score = gini_normalized(truth[:,j], 
                                               preds[:,j], 
                                               sample_weight = sample_weight[:, j])

        except ValueError:
            gini_score = np.NaN
        evaluated_assays[inv_assaymap[j+1]] = {"AUC": roc_score, "auPRC": pr_score, "GINI": gini_score }
            
    return evaluated_assays
