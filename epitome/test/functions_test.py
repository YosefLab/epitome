from epitome.test import EpitomeTestCase
from epitome.test import *
from epitome.functions import *
from epitome.dataset import *
import pytest
import warnings


class FunctionsTest(EpitomeTestCase):

    def __init__(self, *args, **kwargs):
        super(FunctionsTest, self).__init__(*args, **kwargs)

    # def test_download_and_unzip(self):
        # Test on "fake" test dataset

        # Test on wrong non-existing URL
