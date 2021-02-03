from epitome.test import EpitomeTestCase
from epitome.test import *
from epitome.functions import *
from epitome.dataset import *
import pytest
import warnings


class DatasetTest(EpitomeTestCase):

    def __init__(self, *args, **kwargs):
        super(DatasetTest, self).__init__(*args, **kwargs)
        self.dataset = EpitomeDataset()

    def test_user_data_path(self):
        # user data path should be able to be explicitly set
        datapath = GET_DATA_PATH()
        assert(datapath == os.environ["EPITOME_DATA_PATH"])

    def test_save(self):
        out_path = self.tmpFile()
        print(out_path)

        # generate 100 records for each chromosome
        PER_CHR_RECORDS=10
        chroms = []
        newStarts = []
        binSize = 200

        for i in range(1,22): # for all 22 chrs
            chrom = 'chr' + str(i)
            for j in range(1,PER_CHR_RECORDS+1):
                chroms.append(chrom)
                newStarts.append(j * binSize)

        regions_df = pd.DataFrame({'Chromosome': chroms, 'Start': newStarts})

        targets = ['DNase', 'CTCF', 'Rad21', 'SCM3', 'IRF3']
        cellTypes = ['K562'] * len(targets) + ['A549'] * len(targets) + ['GM12878'] * len(targets)
        targets = targets * 3

        row_df = pd.DataFrame({'cellType': cellTypes, 'target': targets})

        # generate random data of 0/1s
        np.random.seed(0)
        rand_data = np.random.randint(2, size=(len(row_df), len(regions_df))).astype('i1')

        source = 'rand_source'
        assembly = 'hg19'
        EpitomeDataset.save(out_path,
                rand_data,
                row_df,
                regions_df,
                binSize,
                assembly,
                source,
                valid_chrs = ['chr7'],
                test_chrs = ['chr8','chr9'])


        # load back in dataset and make sure it checks out
        dataset = EpitomeDataset(data_dir = out_path)

        assert dataset.assembly == assembly
        assert dataset.source == source
        assert np.all(dataset.valid_chrs == ['chr7'])
        assert np.all(dataset.test_chrs == ['chr8','chr9'])
        assert(len(dataset.targetmap)==5)
        assert(len(dataset.cellmap)==3)


    def test_gets_correct_matrix_indices(self):
        eligible_cells = ['IMR-90','H1','H9']
        eligible_targets = ['DNase','H4K8ac']

        matrix, cellmap, targetmap = EpitomeDataset.get_assays(
				targets = eligible_targets,
				cells = eligible_cells, min_cells_per_target = 3, min_targets_per_cell = 1)

        assert(matrix[cellmap['IMR-90']][targetmap['H4K8ac']]==0) # data for first row


    def test_get_assays_single_target(self):
        TF = ['DNase', 'JUND']

        matrix, cellmap, targetmap = EpitomeDataset.get_assays(targets = TF,
                min_cells_per_target = 2,
                min_targets_per_cell = 2)

        targets = list(targetmap)
        # Make sure only JUND and DNase are in list of targets
        assert(len(targets)) == 2

        for t in TF:
            assert(t in targets)

    def test_get_targets_without_DNase(self):
        TF = 'JUND'

        matrix, cellmap, targetmap = EpitomeDataset.get_assays(targets = TF,
                similarity_targets = ['H3K27ac'],
                min_cells_per_target = 2,
                min_targets_per_cell = 1)

        targets = list(targetmap)
        # Make sure only JUND and is in list of targets
        assert(len(targets)) == 2
        assert(TF in targets)
        assert('H3K27ac' in targets)

    def test_list_targets(self):
        targets =self.dataset.list_targets()
        assert len(targets) == len(self.dataset.targetmap)


    def test_get_assays_without_DNase(self):
        TF = 'JUND'

        matrix, cellmap, targetmap = EpitomeDataset.get_assays(targets = TF,
                similarity_targets = ['H3K27ac'],
                min_cells_per_target = 2,
                min_targets_per_cell = 1)

        targets = list(targetmap)
        # Make sure only JUND and is in list of assays
        assert(len(targets)) == 2
        assert(TF in targets)
        assert('H3K27ac' in targets)

    def test_targets_SPI1_PAX5(self):
        # https://github.com/YosefLab/epitome/issues/22
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            matrix, cellmap, targetmap = EpitomeDataset.get_assays(targets = ['DNase','SPI1', 'PAX5'],min_cells_per_target=2, min_targets_per_cell=2)
            assert(len(warning_list) == 1) # one for SPI1 and PAX5
            assert(all(item.category == UserWarning for item in warning_list))


    def test_get_data(self):
        train_data = self.dataset.get_data(Dataset.TRAIN)
        assert train_data.shape[0] == np.where(self.dataset.matrix!= -1)[0].shape[0]
        assert train_data.shape[1] == 1800

        valid_data = self.dataset.get_data(Dataset.VALID)
        assert valid_data.shape[1] == 100

        test_data = self.dataset.get_data(Dataset.TEST)
        assert test_data.shape[1] == 200

        all_data = self.dataset.get_data(Dataset.ALL)
        assert all_data.shape[1] == 2100

        # Make sure you are getting the right data in the right order
        alldata_filtered = self.dataset.get_data(Dataset.ALL)
        dataset = h5py.File(os.path.join(self.dataset.data_dir, EPITOME_H5_FILE), 'r')

        alldata= dataset['data'][:]
        dataset.close()

        assert np.all(alldata[self.dataset.full_matrix[:, self.dataset.targetmap['DNase']],:] == alldata_filtered[self.dataset.matrix[:, self.dataset.targetmap['DNase']],:])

    def test_order_by_similarity(self):
        cell = list(self.dataset.cellmap)[0]
        mode = Dataset.VALID
        sim = self.dataset.order_by_similarity(cell, mode, compare_target = 'DNase')

        assert len(sim) == len(self.dataset.cellmap)
        assert sim[0] == cell

    def test_all_keys(self):
        data = h5py.File(os.path.join(self.dataset.data_dir, EPITOME_H5_FILE), 'r')
        keys = sorted(set(EpitomeDataset.all_keys(data)))

        # bin size should be positive
        assert data['columns']['binSize'][:][0] > 0

        data.close()
        assert np.all([i in REQUIRED_KEYS for i in keys])

    def test_list_genome_assemblies(self):
        assert LIST_GENOME_ASSEMBLIES() == "hg19, test"

    def test_get_data_path(self):
        # Returns env data_path variable when only env data_path var is set
        EpitomeTestCase.setEpitomeDataPath()
        assert GET_DATA_PATH() == os.environ["EPITOME_DATA_PATH"]

        # Fails if both env variables are set
        os.environ[EPITOME_GENOME_ASSEMBLY_ENV] = "test"
        self.assertRaises(AssertionError, GET_DATA_PATH)

        # Returns default data path and genome assembly if only 1 env var is set
        del os.environ[EPITOME_DATA_PATH_ENV]
        assert GET_DATA_PATH() == os.path.join(os.path.join(GET_EPITOME_USER_PATH(), "data"), "test")

        # Clean up test
        del os.environ[EPITOME_GENOME_ASSEMBLY_ENV]
        EpitomeTestCase.setEpitomeDataPath()
