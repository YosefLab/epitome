from epitome.test import EpitomeTestCase
from epitome.test import *
from epitome.functions import *
from epitome.conversion import *
from epitome.dataset import EpitomeDataset
import pytest
import warnings
import pyranges as pr

class ConveresionTest(EpitomeTestCase):

    def __init__(self, *args, **kwargs):
        super(ConveresionTest, self).__init__(*args, **kwargs)

    def test_creates(self):

        base_pr = pr.PyRanges(chromosomes="chr1",
                starts=(100,400,600,800),
                ends=(200,600,800,1000),
                int64=True)

        compare_pr = pr.PyRanges(chromosomes="chr1",
                starts=(400,600,800, 1000,1200),
                ends=(600,800,1000,1200, 1400),
                int64=True)

        conversionObject = RegionConversion(base_pr, compare_pr)
        overlap = conversionObject._get_overlap()
        self.assertTrue(len(overlap) == 3)

        vector, idx = conversionObject.get_binary_vector()
        self.assertTrue(idx.shape[0] == 3)
        self.assertTrue(vector.shape[0] == 4)
        self.assertTrue(np.all(vector == np.array([0,1,1,1])))

        merged = conversionObject.merge(np.array([1,2,3]), axis=0)
        self.assertTrue(merged.shape[0]==5)
        self.assertTrue(np.all(merged[:3] == np.array([1,2,3])))
        self.assertTrue(np.all(np.isnan(merged[3:])))


    def test_2D_3D_merge(self):

        base_pr = pr.PyRanges(chromosomes="chr1",
                starts=(60,70,100,400,1000),
                ends=(65,90,200,600,1200),
                int64=True)

        # first peak overlaps none in base_pr,
        # 2nd overlaps 2, 3rd none, 4th one
        compare_pr = pr.PyRanges(chromosomes="chr1",
                starts=(1,100,700,1100),
                ends=(50, 500,800,1200),
                int64=True)

        conversionObject = RegionConversion(base_pr, compare_pr)

        vector, idx = conversionObject.get_binary_vector()
        self.assertTrue(idx.shape[0] == 3)
        self.assertTrue(vector.shape[0] == len(base_pr))

        matrix = np.ones((4, idx.shape[0], 2)) # 4 samples, 2 TFs

        matrix[0, 0, 0] = 1.5 # a region that will be merged

        axis = 1
        merged = conversionObject.merge(matrix, axis=axis)

        self.assertTrue(merged.shape[axis]==len(compare_pr))
        self.assertTrue(np.all(np.isnan(merged[:,0,:])))
        self.assertTrue(np.all(np.isnan(merged[:,2,:])))
        self.assertTrue(merged[0, 1, 0] == 1.25)
        self.assertTrue(merged[0, 3, 0] == 1)

        self.assertTrue(merged[0, 1, 1] == 1)
        self.assertTrue(merged[0, 3, 1] == 1)



    def test_merge(self):

        base_pr = pr.PyRanges(chromosomes="chr1",
                starts=(1,100,400,600),
                ends=(50, 200,600,800),
                int64=True)

        # 1 peak overlaps 2 regions, the other overlaps none in base_pr
        compare_pr = pr.PyRanges(chromosomes="chr1",
                starts=(100, 900),
                ends=   (500, 1000),
                int64=True)

        conversionObject = RegionConversion(base_pr, compare_pr)

        vector, idx = conversionObject.get_binary_vector()
        self.assertTrue(vector.shape[0] == 4) # shape of base_pr
        vector = np.array([0.5,1]) # only 2 indices should have been scored

        merged = conversionObject.merge(vector, axis=0)
        self.assertTrue(merged.shape[0]==2)
        self.assertTrue(merged[0] == 0.75)
        self.assertTrue(np.isnan(merged[1]))

    def test_score_matrix_combines_indices(self):
        # issue where value_counts() was not sorting on the index,
        # causing predictions to be combined incorrectly and returning preds > 1

        # Create dummy data
        # make 500 regions that do not overlap the Dataset
        start = np.repeat(np.arange(0,100) , 5)
        start = np.concatenate([start,[200,1100,1700]])

        end = np.repeat(np.arange(20,120) , 5)
        end = np.concatenate([end,[900,1500,2100]])

        regions_dict = {'Chromosome': ['chr1'] * len(start),
                        'Start': start,
                        'End': end, 'idx': np.arange(0, start.shape[0])} # only indices 500-502
                                                                         # have data

        regions_pr = pr.from_dict(regions_dict)
        # have to cast to int64
        regions = pr.PyRanges(regions_pr.df, int64=True)

        targets = [ 'CTCF']
        ds = EpitomeDataset(
                targets = targets,
        		cells=['PC-9','Panc1','IMR-90','H1'],
                min_cells_per_target =2,
                data_dir = self.epitome_data_dir,
                assembly=self.epitome_assembly)

        # set predictions to 1s so means could be greater than 1 if done wrong
        preds = np.ones((1, 10, 1))

        conversionObject = RegionConversion(ds.regions, regions)

        results = conversionObject.merge(preds, axis=1)

        masked = np.ma.array(results, mask=np.isnan(results))
        assert(np.all(masked <= 1))


        # Error case where there are nans before true values
        # 1st region on chr 1has no overlap with dataset, while second region
        # on chr2 has multiple (2) overlaps
        start = [30000,200]
        end = [30100,900]
        regions_dict = {'Chromosome': ['chr1','chr2'],
                        'Start': start,
                        'End': end, 'idx': [0,1]}

        regions_pr = pr.from_dict(regions_dict)
        # have to cast to int64
        regions = pr.PyRanges(regions_pr.df, int64=True)

        conversionObject = RegionConversion(ds.regions, regions)

        preds = np.ones((1, 4, 1))

        results = conversionObject.merge(preds, axis=1)
        masked = np.ma.array(results, mask=np.isnan(results))
        assert(np.all(masked <= 1))
