from epitome.test import EpitomeTestCase
from epitome.test import *
from epitome.functions import *
from epitome.dataset import *
import urllib
import os
import pytest
import warnings


class FunctionsTest(EpitomeTestCase):

    def __init__(self, *args, **kwargs):
        super(FunctionsTest, self).__init__(*args, **kwargs)

    def test_download_and_unzip(self):
        # Fails on wrong non-existing URL
        with pytest.raises(urllib.error.HTTPError):
            download_and_unzip("https://epitome-data.s3-us-west-1.amazonaws.com/fake_assembly.zip", self.epitome_test_path)

        # Passes on fake directory without nesting
        test_dir = os.path.join(os.path.dirname(self.epitome_test_path), "fake_dir2")
        download_and_unzip(self.S3_DATA_PATH, test_dir)
