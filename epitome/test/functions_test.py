from epitome.test import EpitomeTestCase
from epitome.test import *
from epitome.functions import *
import pytest
import warnings


class FunctionsTest(EpitomeTestCase):

    def __init__(self, *args, **kwargs):
        super(FunctionsTest, self).__init__(*args, **kwargs)

    def test_user_data_path(self):
        # user data path should be able to be explicitly set
        datapath = GET_DATA_PATH()
        assert(datapath == os.environ["EPITOME_DATA_PATH"])

    def test_get_assays_single_assay(self):
        TF = ['DNase', 'JUND']

        matrix, cellmap, assaymap = get_assays_from_feature_file(eligible_assays = TF,
                min_cells_per_assay = 2,
                min_assays_per_cell = 2)

        assays = list(assaymap)
        # Make sure only JUND and DNase are in list of assays
        assert(len(assays)) == 2

        for t in TF:
            assert(t in assays)

    def test_get_assays_without_DNase(self):
        TF = 'JUND'

        matrix, cellmap, assaymap = get_assays_from_feature_file(eligible_assays = TF,
                similarity_assays = ['H3K27ac'],
                min_cells_per_assay = 2,
                min_assays_per_cell = 1)

        assays = list(assaymap)
        # Make sure only JUND and is in list of assays
        assert(len(assays)) == 2
        assert(TF in assays)
        assert('H3K27ac' in assays)

    def test_assays_SPI1_PAX5(self):
        # https://github.com/YosefLab/epitome/issues/22
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            assays = list_assays()
            matrix, cellmap, assaymap = get_assays_from_feature_file(eligible_assays = assays)

            assert(len(warning_list) == 2) # one for SPI1 and PAX5
            assert(all(item.category == UserWarning for item in warning_list))
