from epitome.test import *
from epitome.functions import *
import pytest

@pytest.fixture
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("data")

def test_download_and_unzip(tmp_dir):

    dirname = os.path.join(tmp_dir.dirname, "sub_dir")
    assert(not os.path.exists(dirname))

    download_and_unzip(S3_TEST_PATH, dirname)

    files = os.listdir(os.path.join(dirname,"data"))
    assert(len(files) == 3)



def test_get_assays_single_assay():
    TF = 'JUND'

    matrix, cellmap, assaymap = get_assays_from_feature_file(feature_name_file = 'epitome/test/data/feature_name',
            eligible_assays = TF,
            min_cells_per_assay = 2,
            min_assays_per_cell = 2)

    assays = list(assaymap)
    # Make sure only JUND and DNase are in list of assays
    assert(len(assays)) == 2
    assert(TF in assays)
    assert('DNase' in assays)
