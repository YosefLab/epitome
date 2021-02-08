r"""
=======================
Vizualization functions
=======================
.. currentmodule:: epitome.viz

.. autosummary::
  :toctree: _generate/

  joint_plot
  plot_assay_heatmap
  heatmap_aggreement_from_model_weights
  calibration_plot
"""

#####################################################################
################### Visualization functions #########################
#####################################################################

try:
    import matplotlib.pyplot as plt
except ImportError:
   import matplotlib
   matplotlib.use('PS')
   import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
from matplotlib.backends import backend_agg
from matplotlib import figure
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

def joint_plot(dict_model1,
               dict_model2,
               metric = "AUC",
               model1_name = "model1",
               model2_name = "model2",
               outlier_filter = None
              ):
    '''
    Returns seaborn joint plot of two models.

    :param dict dict_model1: dictionary of TF: metrics for first model. Output from run_predictions().
    :param doct dict_model2: dictionary of TF: metrics for second model. Output from run_predictions().
    :param dict metric: metric in dicts. Should be auPRC, AUC, or GINI.
    :param str model1_name: string for model 1 name, shown on x axis.
    :param str model2_name: string for model 2 name, shown on y axis.
    :param str outlier_filter: string filter to label. Defaults to no labels.
    :return: matplotlib axis
    :rtype: matplotlib axis
    '''
    sns.set(style="whitegrid", color_codes=True)

    df1 = pd.DataFrame.from_dict(dict_model1).T
    df2 = pd.DataFrame.from_dict(dict_model2).T

    dataframe_joined = pd.concat([df1, df2], axis=1, sort=False)
    # select metric
    dataframe = dataframe_joined[metric]
    dataframe.columns=[model1_name, model2_name]
    dataframe = dataframe.dropna()

    ax = sns.jointplot(x=list(dataframe[model1_name]), y=list(dataframe[model2_name]), kind="reg", stat_func=None)

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



def plot_assay_heatmap(matrix, cellmap, assaymap):
    '''
    Plots a matrix of available assays from available cells. This function takes in the numpy matrix and two dictionaries returned by :code:`get_assays_from_feature_file`.

    :param numpy.matrix matrix: numpy matrix of indices that index into Epitome data
    :param dict cellmap: map of cells indexing into rows of matrix
    :param dict assaymap: map of assays indexing into columns of matrix
    '''

    nv_assaymap = {v: k for k, v in assaymap.items()}

    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.xticks(np.arange(len(assaymap)), rotation = 90)
    ax.set_xticklabels(assaymap.keys())
    plt.yticks(np.arange(len(cellmap)))
    ax.set_yticklabels(cellmap.keys())

    plt.imshow(matrix!=-1)


#####################################################################
############## Network visualization helper functions ###############
#####################################################################

def number_to_bp(n):
    '''
    Converts bp number to short string.

    :param int n: number in base pairs
    :return: string of number with kbp or Mbp suffix
    :rtype: str
    '''
    n = str(n)

    if len(n) < 4:
        return n
    elif len(n) == 4:
        return "%skbp" % n[0]
    elif len(n) == 5:
        return "%skbp" % n[0:1]
    elif len(n) == 6:
        return "%skbp" % n[0:2]
    elif len(n) == 7:
        return "%sMbp" % n[0]
    else:
        raise



def heatmap_aggreement_from_model_weights(model):
    '''
    Plots seaborn heatmap for DNase weights of first layer in network.
    Plots one heatmap for each celltype used in the features for training.

    :param EpitomeModel model: an Epitome model
    '''

    # get weights
    with model.graph.as_default():
        with tf.variable_scope('layer0', reuse=True):
            w = tf.get_variable('kernel')
            weights = model.sess.run(w)

    dnases = weights[len(model.assaymap) * (len(model.cellmap) - 1 - len(model.test_celltypes)): , :]

    num_radii = 2*len(model.radii)

    xtick_labels = [x * 200 for x in model.radii]
    xtick_labels = [number_to_bp(x) for x in xtick_labels]

    for i in range((len(model.cellmap) - 1 - len(model.test_celltypes))):
        # dnases are just at the end
        dnases = x[len(model.assaymap) * (len(model.cellmap) - 2): , :]

        heatmap = dnases[i*num_radii + int(num_radii/2):i*num_radii + num_radii, :].T
        ax = sns.heatmap(heatmap, cmap='bwr_r', vmin=np.min(dnases), vmax=np.max(dnases))
        ax.set_xlabel("Distance from peak (bp)")
        ax.set_ylabel("Unit")
        ax.set_xticklabels(xtick_labels, rotation=-60)
        plt.show()


def calibration_plot(truth, preds, assay_dict, list_assaymap):
    '''
    Creates an xy scatter plot for predicted probability vs true probability.
    Adds a separate set of points for each transcription factor.

    :param numpy.matrix truth: matrix of n samples by t TFs
    :param numpy.matrix preds: matrix same size as truth
    :param dict assay_dict: dictionary of TFs and scores
    :param list list_assaymap: list of assay names for data matrix
    '''

    fig, ax = plt.subplots()
    # only these two lines are calibration curves
    for i in range(truth.shape[1]):
        logreg_y, logreg_x = calibration_curve(truth[:,i], preds[:,i], n_bins=10)

        if (not np.isnan(assay_dict[list_assaymap[i+1]]["AUC"])): # test cell type does not have all factors!
            plt.plot(logreg_x,logreg_y, marker='o', linewidth=1, label=list_assaymap[i+1])


    # reference line, legends, and axis labels
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle('Calibration plot for test regions')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability in each bin')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              ncol=2, fancybox=True, shadow=True)

    plt.show()
