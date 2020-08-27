from epitome.test import *
from epitome.functions import *
import pytest
import warnings

@pytest.fixture
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("data")
        
def test_user_data_path():
    # user data path should be able to be explicitly set
    datapath = GET_DATA_PATH()
    assert(datapath == os.environ["EPITOME_DATA_PATH"])

def test_download_and_unzip(tmp_dir):
    dirname = os.path.join(tmp_dir.dirname, "sub_dir")
    assert(not os.path.exists(dirname))

    download_and_unzip(S3_TEST_PATH, dirname)

    files = os.listdir(os.path.join(dirname,"data"))
    assert(len(files) == 3)


def test_get_assays_single_assay():
    TF = 'JUND'

    matrix, cellmap, assaymap = get_assays_from_feature_file(eligible_assays = TF,
            min_cells_per_assay = 2,
            min_assays_per_cell = 2)

    assays = list(assaymap)
    # Make sure only JUND and DNase are in list of assays
    assert(len(assays)) == 2
    assert(TF in assays)
    assert('DNase' in assays)


def test_assays_SPI1_PAX5():
    # https://github.com/YosefLab/epitome/issues/22
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter('always')
        assays = list_assays()
        matrix, cellmap, assaymap = get_assays_from_feature_file(eligible_assays = assays)

        assert(len(warning_list) == 2) # one for SPI1 and PAX5
        assert(all(item.category == UserWarning for item in warning_list))
        

