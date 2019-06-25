##############################################################
############ Helper functions for picking subsets ############
##############################################################

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
from random import choices
import random

# set seed to give back the same results
random.seed(0)

class Subsetter():
    
    def __init__(self,
                correlation_path,
                matrix,
                assaymap,
                cellmap):
        
        self.corr_pd, self.corr_matrix = self.readCorrelationMatrix(correlation_path)
        self.universe = list(assaymap)[1:] # drop DNase
        self.assaymap = assaymap
        self.cellmap = cellmap
        
        self.binary_matrix = np.copy(matrix)
        self.binary_matrix[self.binary_matrix > -1] = 1
        self.binary_matrix[self.binary_matrix == -1] = 0
        
        
    def readCorrelationMatrix(self, path = '../../data/correlation_matrix.csv'):
        """
        Reads correlation csv file as pandas dataframe.

        :param path: path to correlation_matrix.csv (TODO code for producing this)

        :return pandas dataframe and numpy matrix tuple containing correlation values
        """
        # read correlation matrix
        corr = pd.read_csv(path)
        corr.set_index('TF2', inplace= True)
        corr_matrix = corr.as_matrix().astype(float)
        return corr, corr_matrix

    def plotCorrelationMatrix(self):
        """
        Plots correlation matrix as a heatmap using seaborn.
        """
        ax = sns.heatmap(self.corr_matrix)
        plt.show()

    def get_correlation_score(self, subset):
        """
        Calculates correlation score from a subset of factors, given
        a pre-computed correlation matrix

        :param subset: set of factors
        :param corr: pandas correlation values

        :return mean correlation score
        """

        assert(type(subset) == set), "subset must be of type set"

        if (len(subset) == 1):
            return 1 # always 1 correlation with itself
        corrs = []
        for f in subset:
            corrs.append(np.nanmean(self.corr_pd[f][subset  - {f}]))
        return np.nanmean(corrs)

    def get_subset_score(self, subset):
        """
        Gets subset score from a subset of factors. Calculates score from:
        1. correlation score between factors
        2. number of cells that cover all the factors
        3. size of the subset (larger subsets are easier to train)

        :param subset: set of factors 
        assaymap: map of assays
        cellmap: map of cells:
        binary_matrix: 0/1 np matrix of which cells have data for which factors

        :return score for subset. Note: this value can be NaN. TODO why?
        """

        assert(type(subset) == set), "subset must be of type set"

        # TF correlation scores
        corr_score = self.get_correlation_score(subset)
        # get # cells that have data for all of the factors in this subset
        cell_sums = np.sum(self.binary_matrix[:,[self.assaymap[x] for x in subset]], axis = 1)
        cell_sums_score = np.where(cell_sums == len(subset))[0].shape[0] / len(self.cellmap)

        # equal weighting between subset size, # cells and TF correlation
        final_score = corr_score + cell_sums_score + len(subset)/(len(self.assaymap)-1)
        return final_score


    def random_sample(self, k, n):
        """
        Returns a random sample of k items from universe. Samples n times.

        :param items: list of items to sample from
        :param k: number of items to choose
        :param n: number of times to sample
        """
        choices_ = map(lambda x: choices(self.universe, k = k), range(n))
        choices_ = filter(lambda x: len(set(x)) == k, choices_)
        return list(map(lambda x: (set(x), self.get_subset_score(set(x))), choices_))


    def set_cover(self, subsets):
        """ Greedy weighted set cover search. 
        Finds a family of subsets that covers the universal set. Only iterates 100 times.
        After 100 iterations, adds the remaining singletons not yet covered into the final cover.

        :param universe: set of factors
        :param subset: list of tuples of (subset, score) from where subset is from universe

        :return cover: returns 
        """
        elements = set(e for s in subsets for e in s[0])
        # Check the subsets cover the universe
        if sorted(elements) != sorted(self.universe):
            return None
        covered = set()
        cover = []
        iterations = 0
        # Greedily add the subsets with the most uncovered points
        while sorted(covered) != sorted(elements) and iterations < 100:
            iterations = iterations + 1
            subset = max(subsets, key=lambda s: s[1] * len(s[0] - covered))

            # only add if there is a new TF in this subset
            if len(subset[0] - covered) > 0:
                cover.append(subset[0])
                covered |= subset[0]

        print("Missing %s. Adding as single models..." % (elements - covered))

        for i in elements - covered:
            cover.append({i})

        return cover
