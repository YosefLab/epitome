from epitome.test import EpitomeTestCase
from epitome.test import *
from epitome.functions import *
from epitome.dataset import *
import pytest
import warnings


class FunctionsTest(EpitomeTestCase):

    def __init__(self, *args, **kwargs):
        super(FunctionsTest, self).__init__(*args, **kwargs)

    def test_user_data_path(self):
        # user data path should be able to be explicitly set
        datapath = GET_DATA_PATH()
        assert(datapath == os.environ["EPITOME_DATA_PATH"])

    def test_pyranges_intersect(self):

        dataset = EpitomeDataset()

        pr1 = dataset.regions.head(10)
        pr2 = dataset.regions
        res = pyranges2Vector(pr1, pr2)

        assert np.all(res[0][:10] == True)
        assert np.all(res[0][10:] == False)
        assert res[1][1].shape[0] == len(pr1)
        assert len(res[1][0]) == len(pr1)
