#####################################################################
################### Visualization functions #########################
#####################################################################

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def joint_plot(dict_model1, 
               dict_model2, 
               metric = "AUC", 
               model1_name = "model1", 
               model2_name = "model2",
               outlier_filter = None):
    """
    Returns seaborn joint plot of two models.
    
    :param dict_model1: dictionary of TF: metrics for first model. Output from run_predictions().
    :param dict_model2: dictionary of TF: metrics for second model. Output from run_predictions().
    :param metric: metric in dicts. Should be auPRC, AUC, or GINI.
    :param model1_name: string for model 1 name, shown on x axis.
    :param model2_name: string for model 2 name, shown on y axis.
    :param outlier_filter: string filter to label. Defaults to no labels.
    """
    df1 = pd.DataFrame.from_dict(dict_model1).T
    df2 = pd.DataFrame.from_dict(dict_model2).T

    dataframe_joined = pd.concat([df1, df2], axis=1, sort=False)
    # select metric
    dataframe = dataframe_joined[metric]
    dataframe.columns=[model1_name, model2_name]

    ax = sns.jointplot(model1_name, model2_name, data=dataframe, kind="reg", color='k', stat_func=None)
    ax.ax_joint.cla()
    ax.set_axis_labels(model1_name, model2_name)
    ax.fig.suptitle(metric) 

    def ann(row):
        ind = row[0]
        r = row[1]
        plt.gca().annotate(ind, xy=(r[model1_name], r[model2_name]), 
                xytext=(2,2) , textcoords ="offset points", )
        
    # filter for labels
    if (outlier_filter is not None):
        label_df = dataframe.query(outlier_filter)

        for index, row in label_df.iterrows():
            if (not np.isnan(row[model1_name]) and not np.isnan(row[model2_name])):
                ann((index, row))

    for i,row in dataframe.iterrows():
        color = "blue" if row[model1_name] <  row[model2_name] else "red"
        ax.ax_joint.plot(row[model1_name], row[model2_name], color=color, marker='o')
        
    return ax
