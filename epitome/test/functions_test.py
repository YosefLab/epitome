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