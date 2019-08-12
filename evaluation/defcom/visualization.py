import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
import seaborn as sns
from os.path import join

#####################################################################
################# Visualization and Plots ###########################
#####################################################################


def generateComparativeBoxplot(eval_results, metric, save_plot_name=None):
    """
    Generates a comparative boxplot based on the evaluation results. It will save the plot to a file if the save plot name
    is specified
    
    :param eval_results: dataframe containing eval_results, its column names should be as follows:
        model | training_cell | transcription_factor | query_cell | auROC | auPR
        the reason it is passed as a dataframe is to allow the user to make any changes before plotting
        such as excluding certain tfs.        
    :param metric: what evaluation metric to be plotted, should either be 'auROC' for area under receiver
        operating characteristic curve or 'auPR' for area under the precision recall curve.        
    :param save_plot_name: optional output path to where the file should be saved to. Should end in .png
    
    :return None but outputs a comparative boxplot and saves to a file if save_plot_name is specified
    """
    assert(metric == 'auPR' or metric == 'auROC') , \
        "The metric specified must either be 'auROC' for area under the ROC curve or 'auPR' for area under the PR curve"
    # make the boxplot
    g = sns.catplot(x="model", y=metric, col='query_cell', kind="box", data=eval_results)
    g.set_xticklabels(rotation=45)
    g.fig.suptitle('Performance Comparison Epitome vs DeFCoM')
    g.fig.set_size_inches(15,15)
    
    # save the boxplot if specififed
    if(save_plot_name != None):
        g.savefig(save_plot_name)       
        
        
        
def generateScatterPlots(results, metric, save_plot_dir=None):
    """
    Generates a scatterplot based on the evaluation results. It will save the plots to a folder if the param
        save_plot_dir is specififed
    
    :param results: dataframe containing the results, its column names should be as follows:
        model | training_cell | transcription_factor | query_cell | auROC | auPR
        the reason it is passed as a dataframe is to allow the user to make any changes before plotting
        such as excluding certain tfs.
        
    :param save_plot_dir: optional output path to a directory where the plots should be saved to. Should be a directory
        
    :return None but either prints a scatterplot or saves scatterplots to a directory if save_plot_dir is specified
    """
    assert(metric == 'auPR' or metric == 'auROC') , \
        "The metric specified must either be 'auROC' for area under the ROC curve or 'auPR' for area under the PR curve"

    plt.style.use('ggplot')

    # list of markers and colors
    markers = {'k562':    'o',
               'hepg2':   '^',
               'h1hesc':  's',
               'gm12878': 'x',
               'joint':    '*'}
    colors = {'defcom' :  'r',
              'epitome' : 'b'}
    cells = ['k562', 'gm12878', 'hepg2', 'h1hesc']#, 'joint']
    epitome_cells = [('K562', 'k562'),
                ('GM12878', 'gm12878'),
                ('HepG2', 'hepg2'),
                ('H1-hESC', 'h1hesc')]

    results['training_cell'] = results['training_cell'].apply(lambda x: x.lower().replace('-',''))
    results['query_cell'] = results['query_cell'].apply(lambda x: x.lower().replace('-',''))

    #results['training_cell'] = results['training_cell'].lower().replace('-','')
    #results['query_cell'] = results['query_cell'].lower().replace('-','')
    # get the query results
    query_cell_results = []

    for cell in cells:
        query_cell_results.append(results.loc[(results['query_cell'] == cell)])
    # iterate through each cell type
    for query_cell_result in query_cell_results:
        # plot each point
        for index, result in query_cell_result.iterrows():
            model = result['model'] # get the model
            training_cell = result['training_cell'] # get the training cell
            query_cell = result['query_cell'] # get the query cell
            tf = result['transcription_factor'].lower() # should make them all lowercase
            auROC = result['auROC']
            auPR = result['auPR']

            # account for missing tfs or differences in names...
            '''
            if(tf == 'c-myc'):
                tf = 'myc'
            if(tf == 'ep300' or tf == 'gabpa' or tf == 'rest'):
                continue
            '''

            # get plot details
            mi = markers[training_cell]
            #ci = colors[model]
            ci = 'r' if 'defcom' in model else 'b' # instead of indexing 
            xi = tf
            yi = auPR if metric == 'auPR' else auROC
            # plot points
            plt.scatter(xi, yi, marker=mi, color=ci)

        # create the legend
        pltHandles = [mlines.Line2D([], [], color='black', marker=cellMarker, linestyle='None',
                                 markersize=10, label=cellLabel) for cellLabel, cellMarker in markers.items()]
        pltHandles = pltHandles + [mlines.Line2D([], [], color=modelColor, marker='o', linestyle='None',
                           markersize=10, label=modelLabel) for modelLabel, modelColor in colors.items()]

        # label plot
        query_cell_name = query_cell_result['query_cell'].values[0]
        plotTitle = 'Query Cell: {} Prediction Results'.format(query_cell_name)
        plt.title(plotTitle)
        lgd = plt.legend(handles=pltHandles, loc='center left', ncol=1, fancybox=True, bbox_to_anchor=(1, 0.5), shadow=True)
        plt.xlabel('Transcription Factor')
        ylabel = 'Area Under PR Curve' if metric == 'auPR' else 'Area Under ROC Curve'
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)    

        # save the plot
        if(save_plot_dir != None):
            plot_name = '{}_{}_scatterplot_plot.png'.format(query_cell_name, metric)
            plotSaveFile = join(save_plot_dir,plot_name)
            plt.savefig(plotSaveFile, bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            plt.show()
        
        plt.clf()