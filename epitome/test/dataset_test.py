from epitome.test import EpitomeTestCase
import epitome
import numpy as np
import pytest

@pytest.fixture
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("epitome")

class DatasetTest(EpitomeTestCase):
    def __init__(self, *args, **kwargs):
        super(EpitomeTestCase, self).__init__(*args, **kwargs)

    def test_LIST_GENOMES(self):
        assert(LIST_GENOMES() == 'hg19')
        
#     def test_GET_EPITOME_USER_PATH(self):
#         tmp_dir()
#         assert()
    
    def test_GET_DATA_PATH(self):
#         tmp_data_dir = "epitome/data/test"
        
        # Returns env data_path variable when only env data_path var is set
#         os.environ[EPITOME_DATA_PATH_ENV] = tmp_data_dir
        assert(GET_DATA_PATH() == os.environ["EPITOME_DATA_PATH"])

        # Fails if both env variables are set
        os.environ[EPITOME_GENOME_ASSEMBLY_ENV] = "test"
        self.assertRaises(AssertionError, GET_DATA_PATH())
        
        # Returns default data path and genome assembly if only 1 env var is set
        del os.environ[EPITOME_DATA_PATH_ENV]
        assert(GET_DATA_PATH() == os.path.join(os.path.join(GET_USER_PATH(), "data"), "test"))