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
    
    def test_compute_casv(self):
        a = np.ones((5, 1, 4))
        g = np.zeros((5, 4))

        out = compute_casv(g, a, [1])

        assert out.shape == (5, 1, 4, 4)
        assert np.all(out == 0)
