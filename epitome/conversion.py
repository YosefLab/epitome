r"""
==========
Conversion
==========
.. currentmodule:: epitome.conversion

.. autosummary::
  :toctree: _generate/

  RegionConversion
"""

from .functions import bed2Pyranges
import pyranges as pr
import numpy as np

class RegionConversion:
    '''
    Class for dealing with genomic region conversions. In Epitome,
    we often have to join a user's query genomic regions with a larger
    region set (i.e. EpitomeDataset regions). This class performs two functions:
    converting a user defined list of genomic regions to a vector that matches
    a larger bed file, and converting Epitome predictions back to the original bed
    regions.

    '''

    def __init__(self, base_bed, compare_bed):
        '''
        Initialization function for RegionConversion class.

        :param str|pyranges base_bed: either path to bed file or indexed pyranges base object
        :param str|pyranges compare_bed: either path to bed file or indexed pyranges comparison object
        '''

        # load in bed files as pyranges with index column
        # types should end up being pyranges
        self.base = RegionConversion.convert(base_bed)
        self.compare = RegionConversion.convert(compare_bed)
        self.joined = self.compare.join(self.base, how='left',suffix='_base')

    @staticmethod
    def convert(regions):
        '''
        Converts a bed file to indexed pyranges. If already a pyranges object, returns.

        :param str|pyranges regions: either path to bed file or indexed pyranges object
        :return: indexed PyRanges
        :rtype: pr.pyranges
        '''

        if type(regions) == str:
            regions_bed = bed2Pyranges(regions)
        elif type(regions) == pr.PyRanges:
            if 'idx' not in regions.columns:
                regions.idx = np.arange(0, len(regions))
                print("Warning: regions pyranges must have column named 'idx'")
            regions_bed = regions
        else:
            raise Exception("regions must be type scoring or pr.Pyranges, but got type %s" % type(regions))
        return regions_bed


    def _get_overlap(self):
        '''
        Returns locations where there is supporting data in
        both base and compare

        :return: pyranges object containing regions that have overlapping data
        for both compare and base
        :rtype: pr.pyranges
        '''
        return self.joined[self.joined.idx_base > -1]

    def get_base_overlap_index(self):
        '''
        Returns index of locations in *base* where there is supporting data in
        both base and compare
        '''
        return self._get_overlap().idx_base


    def get_binary_vector(self, vector = None):
        '''
        Finds indices in base that overlap compare and return vector matching
        regions.

        :return: tuple of vector of len(base) with ones that match compare, and
            indices in base that have data for compare
        :rtype: tuple
        '''

        if vector is None:
            vector = np.ones(len(self.compare))

        assert len(vector.shape) == 1, "Error: value_vector must be a 1D array"
        assert vector.shape[0] == len(self.compare), "Error: value_vector must be the same shape as self.compare"

        base_vector = np.zeros((len(self.base)))
        tmp = self._get_overlap()

        base_indices = tmp.idx_base.values
        convert_indices = tmp.idx.values

        base_vector[base_indices] = vector[convert_indices]

        return base_vector, base_indices

    def compare_df(self):
        ''' Gets genomic regions ordered by idx '''
        return self.compare.df.sort_values(by='idx').drop(labels='idx', axis=1)

    def merge(self, matrix, axis = -2):
        '''
        Groups matrix shaped by base pyranges and calculates new predictions by
        taking the mean to match the shape of compare pyranges.
        This is for the case when you are predicting in regions
        that may overlap multiple regions in an EpitomeDataset.

        :param np.matrix matrix: matrix, where exactly 1 axis matches the shape of joined base/compare regions
        :param int axis: axis that represents genomic regions. Defaults to -2.

        :return: reduced np.matrix, axis specified is reduced
        :rtype: np.matrix
        '''

        joined = self._get_overlap() \
            .df \
            .sort_values(by='idx') \
            .reset_index(drop=True)

        assert matrix.shape[axis] == len(joined), """joined and matrix must have the same dimension,
            but got %i and %i """ % (matrix.shape[axis],len(joined))

        # get the index break for each region_bed region
        deduped = joined.drop_duplicates('idx',keep='first')
        reduce_indices = deduped.index.values

        # get the number of times there was a scored region for each region_bed region
        # used to calculate reduced means
        indices_counts = joined['idx'].value_counts(sort=False).sort_index().values


        # define shape for results. make sure regions axis is length of compare.
        shape = list(matrix.shape)
        shape[axis] = len(self.compare)
        final = np.empty((shape))
        final[:] = np.nan

        for i in range(len(matrix.shape)):
            if i != axis:
                indices_counts = np.expand_dims(indices_counts, axis=i)

        reduced_values = np.add.reduceat(matrix, reduce_indices, axis = axis)/indices_counts

        # get index to select and expand dimensions to match final
        index = deduped['idx'].values

        for i in range(len(final.shape)):
            if i != axis:
                index = np.expand_dims(index, axis=i)

        # put reduced slices in correct indices
        np.put_along_axis(final, index, reduced_values, axis)

        return final
