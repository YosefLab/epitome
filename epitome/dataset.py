r"""
==============
EpitomeDataset
==============
.. currentmodule:: epitome.dataset

.. autosummary::
  :toctree: _generate/

  EpitomeDataset
"""


import pyranges as pr
import h5py
import numpy as np
import os
import pandas as pd
from collections import Counter
import warnings
from sklearn.metrics import jaccard_score

# local imports
from epitome import *
from .constants import *
from .functions import download_and_unzip
from .viz import plot_assay_heatmap

class EpitomeDataset:
    '''
    Dataset for holding Epitome data.
    Data processing scripts can be found in epitome/data.

    '''

    def __init__(self,
                 data_dir=None,
                 targets = None,
                 cells = None,
                 min_cells_per_target = 3,
                 min_targets_per_cell = 2,
                 similarity_targets = ['DNase']):
        '''
        Initializes an EpitomeDataset.

        :param str data_dir: path to data directory containing data.h5.
          By default, accesses data in ~/.epitome/data
        :param list targets: list of ChIP-seq targets to include in dataset
        :param list cells: list of celltypes to use in dataset
        :param int min_cells_per_target: minimum number of cell types required for
          a given ChIP-seq target
        :param int min_targets_per_cell: minimum number of ChIP-seq targets required
          for each celltype
        :param list similarity_targets: list of targets to be used to compute similarity
          (ie. DNase, H3K27ac, etc.)
        '''


        # get directory where h5 file is stored
        self.data_dir = EpitomeDataset.get_data_dir(data_dir)
        self.h5_path = os.path.join(self.data_dir, EPITOME_H5_FILE)

        # save all parameters for any future use
        self.targets = targets
        self.cells = cells
        self.min_cells_per_target = min_cells_per_target
        self.min_targets_per_cell =min_targets_per_cell
        self.similarity_targets = similarity_targets

        # load in specs for data
        self.full_matrix, self.cellmap, self.targetmap = EpitomeDataset.get_assays(targets = targets,
                                     cells = cells,
                                     data_dir = self.data_dir,
                                     min_cells_per_target = self.min_cells_per_target,
                                     min_targets_per_cell = self.min_targets_per_cell,
                                     similarity_targets = similarity_targets)


        # make a truncated matrix that includes updated indices for rows containing data from cellmap, targetmap
        self.matrix = self.full_matrix.copy()
        self.row_indices = self.full_matrix.flatten()
        self.row_indices=self.row_indices[self.row_indices!=-1]
        # update matrix values
        for i,v in enumerate(self.row_indices):
            self.matrix[self.full_matrix == v] = i

        # set similarity targets and list of targets to be predicted
        self.similarity_targets = similarity_targets
        self.predict_targets = list(self.targetmap)
        [self.predict_targets.remove(i) for i in self.similarity_targets]


        # read in dataset
        dataset = h5py.File(self.h5_path, 'r')
        keys = EpitomeDataset.all_keys(dataset)

        # make sure dataset has all required keys
        assert np.all([i in keys for i in REQUIRED_KEYS ]), "Error: missing required keys in dataset at %s " % self.h5_path

        # where data will be stored
        # this will be loaded lazily as the user needs them
        self._data = None


        # Load in genomic regions as pyranges object
        # NOTE: pyranges sorts the chrommosomes by default, so we use an idx column to
        # indicate the column in the dataset. We use pyranges instead of a pandas dataframe
        # because it uses much less memory.
        self.regions = pd.DataFrame({'Chromosome':dataset['columns']['chr'][:].astype(str),
           'Start':dataset['columns']['start'][:],
            'End':dataset['columns']['start'][:] + dataset['columns']['binSize']})

        self.regions['idx']=self.regions.index
        self.regions = pr.PyRanges(self.regions, int64=True)

        # save indices for later use
        self.indices = {}
        self.indices[Dataset.TRAIN] = dataset['columns']['index'][Dataset.TRAIN.name][:]
        self.indices[Dataset.VALID] = dataset['columns']['index'][Dataset.VALID.name][:]
        self.indices[Dataset.TEST] = dataset['columns']['index'][Dataset.TEST.name][:]
        self.valid_chrs = [i.decode() for i in dataset['columns']['index']['valid_chrs'][:]]
        self.test_chrs = [i.decode() for i in dataset['columns']['index']['test_chrs'][:]]

        self.assembly = dataset['meta']['assembly'][:][0].decode()
        self.source = dataset['meta']['source'][:][0].decode()


        dataset.close()

    def get_parameter_dict(self):
        ''' Returns dict of all parameters required to reconstruct this dataset

        :return: dict containing all parameters to reconstruct dataset.
        :rtype: dict
        '''

        return {'data_dir':self.data_dir,
                'targets': self.targets,
                'cells': self.cells,
                'min_cells_per_target': self.min_cells_per_target,
                'min_targets_per_cell': self.min_targets_per_cell,
                'similarity_targets': self.similarity_targets}

    def get_data(self, mode):
        '''
        Lazily loads all data into memory.

        :param Dataset enum mode: Dataset enumeration. Dataset.TRAIN, Dataset.TEST, Dataset.VALID, or Dataset.ALL

        :return: self._data for a given mode
        :rtype: numpy.matrix
        '''

        if self._data is None:
            dataset = h5py.File(self.h5_path, 'r')

            # rows need to be sorted first before accessing
            order = np.argsort(self.row_indices)
            i = np.empty_like(order)
            i[order] = np.arange(order.size)

            # Indexing load time is about 1s per row.
            # Because it takes about 1min to load all of the data into memory,
            # it is just quicker to load all data into memory when you are accessing
            # more than 100 rows.
            if order.shape[0] > 60:
                # faster to just load the whole thing into memory then subselect
                self._data = dataset['data'][:,:][self.row_indices[order],:][i,:]
            else:
                self._data = dataset['data'][self.row_indices[order],:][i,:]

            dataset.close()

        if mode == Dataset.ALL:
            return self._data
        else:
            return self._data[:,self.indices[mode]]


    @staticmethod
    def get_y_indices_for_cell(matrix, cellmap,  cell):
        '''
        Gets indices for a cell.
        TODO: this function is called in genertors.py.
        Once generators.py is rebased to use dataset,
        this function should NOT be static.

        :param str cell: celltype name

        :return: locations of indices for the cell name specified
        :rtype: numpy.array
        '''

        return np.copy(matrix[cellmap[cell],:])

    @staticmethod
    def get_y_indices_for_target(matrix, targetmap, target):
        '''
        Gets indices for a assay.
        TODO: this function is called in genertors.py.
        Once generators.py is rebased to use dataset,
        this function should NOT be static.

        :param str target: str target
        :return: locations of indices for the cell name specified
        :rtype: numpy.array
        '''
        return np.copy(matrix[:,targetmap[target]])

    @staticmethod
    def get_data_dir(data_dir=None):
        '''
        Loads data processed from data/download_encode.py. This will check that all required files
        exist.

        :param str data_dir: Directory containing data.h5 file saved in data/download_encode.py script.
        :return: directory containing data.h5 file
        :rtype: str
        '''

        if not data_dir:
            data_dir = GET_DATA_PATH()
            download_and_unzip(S3_DATA_PATH, os.path.dirname(data_dir))

        # make sure all required files exist
        required_paths = [os.path.join(data_dir, x) for x in  REQUIRED_FILES]
        assert(np.all([os.path.exists(x) for x in required_paths]))

        return data_dir

    def list_targets(self):
        '''
        Returns available ChIP-seq targets/chromatin accessibility targets
        available in the curretn dataset.

        :return: list of target names
        :rtype str
        '''
        return list(self.targetmap)


    @staticmethod
    def get_assays(targets = None,
                     cells = None,
                     data_dir = None,
                     min_cells_per_target = 3,
                     min_targets_per_cell = 2,
                     similarity_targets = ['DNase']):
        '''
        Returns at matrix of cell type/targets which exist for a subset of cell types.

        :param list targets: list of targets to filter by (ie ["CTCF", "EZH2", ..]). If None, then returns all targets.
        :param list cells: list of cells to filter by (ie ["HepG2", "GM12878", ..]). If None, then returns all cell types.
        :param str data_dir: path to data. should have data.h5 here
        :param int min_cells_per_target: number of cell types an target must have to be considered
        :param int min_targets_per_cell: number of targets a cell type must have to be considered. Includes DNase.
        :param list similarity_targets: target to use for computing similarity
        :return: matrix: cell type by target matrix
                cellmap: index of cells
                targetmap: index of targets
        :rtype: tuple
        '''

        if not data_dir:
            data_dir = GET_DATA_PATH()

        data = h5py.File(os.path.join(data_dir, EPITOME_H5_FILE), 'r')

        # check argument validity
        if (min_targets_per_cell < 2):
             warnings.warn("min_targets_per_cell should not be < 2 (this means it only has a similarity target) but was set to %i" % min_targets_per_cell)

        if (min_cells_per_target < 2):
             warnings.warn("min_cells_per_target should not be < 2 (this means you may only see it in test) but was set to %i" % min_cells_per_target)

        if (targets != None):

            # make sure eligible targets is a list, and not a single target
            if type(targets) == str:
                targets = [targets]

            # similarity targets must be in the list
            for a in similarity_targets:
              if a not in targets:
                targets = [a] + targets

            if (len(targets) + 1 < min_targets_per_cell):
                raise Exception("""%s is less than the minimum targets required (%i).
                Lower min_targets_per_cell to (%i) if you plan to use only %i eligible targets""" \
                                % (targets, min_targets_per_cell, len(targets)+1, len(targets)))

        if (cells != None):
            if (len(cells) + 1 < min_cells_per_target):
                raise Exception("""%s is less than the minimum cells required (%i).
                Lower min_cells_per_target to (%i) if you plan to use only %i eligible cells""" \
                                % (cells, min_cells_per_target, len(cells)+1, len(cells)))


        # Want a dictionary of target: {list of cells}
        # then filter out targets with less than min_cells_per_target cells
        # after this, there may be some unused cells so remove those as well

        indexed_targets={}    # dict of {cell: {dict of indexed targets} }
        for i, (cell, target) in enumerate(zip(data['rows']['celltypes'][:], data['rows']['targets'][:])):

            # bytes to str
            cell, target = cell.decode(), target.decode()

            # check if cell and target is valid
            valid_cell = (cells == None) or (cell in cells)
            valid_target = (targets == None) or (target in targets)

            # if cell and target is valid, add it in
            if valid_cell and valid_target:
                if cell not in indexed_targets:
                    indexed_targets[cell] = {target: i}
                else:
                    indexed_targets[cell][target] = i



        # finally filter out cell types with < min_targets_per_cell and have data for similarity_targets
        indexed_targets = {k: v for k, v in indexed_targets.items() if np.all([s in v.keys() for s in similarity_targets]) and len(v) >= min_targets_per_cell}

        # make flatten list of targets from cells
        tmp = [list(v) for k, v in indexed_targets.items()]
        tmp = [item for sublist in tmp for item in sublist]

        # list of targets that meet min_cell criteria
        valid_targets = {k:v for k, v in Counter(tmp).items() if v >= min_cells_per_target}

        # remove invalid targets from indexed_targets
        for key, values in indexed_targets.items():

            # remove targets that do not mean min_cell criteria
            new_v = {k: v for k, v in values.items() if k in valid_targets.keys()}
            indexed_targets[key] = new_v

        potential_targets = valid_targets.keys()
        cells_dict = indexed_targets.keys()

        # sort cells alphabetical
        cells_dict = sorted(cells_dict, reverse=True)

        # sort targets alphabetically
        potential_targets = sorted(potential_targets, reverse=True)

        cellmap = {cell: i for i, cell in enumerate(cells_dict)}
        targetmap = {target: i for i, target in enumerate(potential_targets)}

        matrix = np.zeros((len(cellmap), len(targetmap))) - 1
        for cell in cells_dict:
            for target, _ in indexed_targets[cell].items():
                matrix[cellmap[cell], targetmap[target]] = indexed_targets[cell][target]

        # finally, make sure that all targets that were specified are in targetmap
        # if not, throw an error and print the reason.
        if targets is not None:

            missing = [i for i in targets if i not in list(targetmap)]
            for a in missing:
                warnings.warn('%s does not have enough data for cutoffs of min_cells_per_target=%i and min_targets_per_cell=%i' %
                              (a, min_cells_per_target, min_targets_per_cell))

        matrix = matrix.astype(int)
        return matrix, cellmap, targetmap

    @staticmethod
    def all_keys(obj, keys=[]):
        '''
        Recursively find all keys in an openh5py dataset

        :param h5py.Group obj: h5py group to recurse
        :param list keys: list of keys to returns
        :return: list of keys
        :rtype: list
        '''
        keys.append(obj.name)
        if isinstance(obj, h5py.Group):
            for item in obj:
                if isinstance(obj[item], h5py.Group):
                    EpitomeDataset.all_keys(obj[item], keys)
                else: # isinstance(obj[item], h5py.Dataset):
                    keys.append(obj[item].name)
        return keys

    def order_by_similarity(self, cell, mode, compare_target = 'DNase'):
        '''
        Orders list of cellmap names by similarity to comparison cell.

        :param str cell: name of cell type, should be in cellmap
        :param Dataset mode: Dataset mode to select data.
        :param str compare_target: target to use to compare cell types. Default = DNase
        :return: list of cellline names ordered by DNase similarity to cell (most similar is first)
        :rtype: list
        '''

        data = self.get_data(mode)

        # data for cell line to compare all other cell lines to
        compare_arr = data[self.matrix[self.cellmap[cell], self.targetmap[compare_target]],:]


        # calculate jaccard score
        corrs = np.array([jaccard_score(data[self.matrix[self.cellmap[c],
                                                                     self.targetmap[compare_target]],:], compare_arr) for c in list(self.cellmap)])

        tmp = sorted(zip(corrs, list(self.cellmap)), key = lambda x: x[0], reverse=True)
        return list(map(lambda x: x[1],tmp))

    @staticmethod
    def save(out_path,
            all_data,
            row_df,
            regions_df,
            binSize,
            assembly,
            source,
            valid_chrs = ['chr7'],
            test_chrs = ['chr8','chr9']):
        ''' Saves an Epitome dataset.

        :param str out_path: directory to save data.h5 file to
        :param numpy.matrix all_data: binary numpy matrix of shaple (len(row_df), len(regions_df))
        :param pandas.dataframe row_df: dataframe containing row information. Should have column names "cellType" and "target"
        :param pandas.dataframe regions_df: dataframe containing column genomic regions. Should have column names
            ['Chromosome', 'Start']. End-Start should always be the same width (200bp or so).
        :param int binSize: size of each genomic region in regions_df.
        :param list valid_chrs: list of validation chromsomes, str
        :param list test_chrs: list of test chromsomes, str

        '''

        if os.path.exists(os.path.join(out_path, EPITOME_H5_FILE)):
            raise Exception("%s already exists at %s" % (EPITOME_H5_FILE, out_path))

        # assertions
        assert all_data.dtype == np.dtype('int8'), "all_data type should be int8"
        assert len(regions_df) == all_data.shape[1], "all_data columns must equal len(regions_df)"
        assert len(row_df) == all_data.shape[0], "all_data rows must equal len(row_df)"
        assert np.all([i in row_df for i in ["cellType","target"]]), "row_df is missing required columns cellType, target"
        assert np.all([i in regions_df for i in ['Chromosome', 'Start']]), "regions_df is missing required columns Chromosome,Start,End"
        assert type(assembly) == str, "assembly must be type string"
        assert type(source) == str, "source must be type string"
        assert np.all([type(i)==str for i in valid_chrs]), "valid_chrs elements must be type string"
        assert np.all([type(i)==str for i in test_chrs]), "test_chrs elements must be type string"

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        try:
            # toy dataset with everything in it
            new_data = h5py.File(os.path.join(out_path, EPITOME_H5_FILE), 'w')

            # 0. set data
            data = new_data.create_dataset("data", all_data.shape, dtype=all_data.dtype, compression="gzip", compression_opts=9)
            data[:,:] = all_data

            # 1. make row info
            rows = new_data.create_group("rows")

            # assign row celltypes
            cellList = list(row_df['cellType'])
            max_len = max([len(i) for i in cellList])
            celltypes = rows.create_dataset("celltypes", (len(cellList),),dtype="|S%i" % max_len,
                                            compression="gzip", compression_opts=9, shuffle=True)

            for i, c in enumerate(cellList):
                celltypes[i]=c.encode()

            # assign row targets
            targetList = list(row_df['target'])
            max_len = max([len(i) for i in targetList])
            targets = rows.create_dataset("targets", (len(targetList),),dtype="|S%i" % max_len,
                                          compression="gzip", compression_opts=9, shuffle=True)

            for i, c in enumerate(targetList):
                targets[i]=c.encode()

            # 2. make column info
            cols = new_data.create_group("columns")

            chrs = cols.create_dataset("chr", (len(regions_df),), dtype="|S5", # S5 = chrXX
                                       compression="gzip", compression_opts=9)
            start = cols.create_dataset("start", (len(regions_df),),
                                        dtype="i",compression="gzip", compression_opts=9)

            bs = cols.create_dataset("binSize", (1,), dtype="i")
            bs[:]=binSize

            tmp = [c.encode() for c in regions_df['Chromosome'].values]
            chrs[:]=tmp
            start[:]= regions_df['Start'].values

            # 3. Make index that specifies where train, valid, and test indices start.
            index = cols.create_group("index")

            # cast valid/test chrs to bytes
            valid_chrs = [i.encode() for i in valid_chrs]
            test_chrs = [i.encode() for i in test_chrs]

            # save test and valid chrs
            vcs = index.create_dataset("valid_chrs", (len(valid_chrs),), dtype="|S5", # S5 = chrXX
                                       compression="gzip", compression_opts=9)
            vcs[:]=valid_chrs

            tcs = index.create_dataset("test_chrs", (len(test_chrs),), dtype="|S5", # S5 = chrXX
                                       compression="gzip", compression_opts=9)
            tcs[:]=test_chrs

            valid_indices = np.where(np.isin(chrs[:], valid_chrs))[0]
            test_indices = np.where(np.isin(chrs[:], test_chrs))[0]
            assert len(valid_indices)>0
            allindices = np.arange(0, len(regions_df))
            train_indices = np.where(~np.isin(allindices, np.concatenate([valid_indices, test_indices])))[0]

            train = index.create_dataset(Dataset.TRAIN.name, (train_indices.shape[0],), dtype="i",
                                       compression="gzip", compression_opts=9)
            valid = index.create_dataset(Dataset.VALID.name, (valid_indices.shape[0],),
                                        dtype="i",compression="gzip", compression_opts=9)
            test = index.create_dataset(Dataset.TEST.name, (test_indices.shape[0],),
                                        dtype="i",compression="gzip", compression_opts=9)

            train[:] = train_indices
            valid[:]= valid_indices
            test[:] = test_indices

            # 4. add metadata
            meta = new_data.create_group("meta")
            assembly_ds = meta.create_dataset('assembly', (1,), dtype="|S%i" % len(assembly),
                                       compression="gzip", compression_opts=9)
            assembly_ds[:]=assembly.encode()
            source_ds = meta.create_dataset('source', (1,), dtype="|S%i" % len(source),
                                       compression="gzip", compression_opts=9)
            source_ds[:]=source.encode()


            # 4. Make sure we have all the correct keys
            keys = sorted(set(EpitomeDataset.all_keys(new_data)))
            assert np.all([i in keys for i in REQUIRED_KEYS ]), "Error: missing required keys for new dataset"

            new_data.close()


        finally:
            # always close resources
            new_data.close()

    def saveToyData(self, toy_path):
            '''
            Creates a toy dataset for test from this dataset.

            Copies over targets and cells, then generates synthetic regions and matrix
            for 22 chrs

            :param str toy_path: path to save toy dataset to.
            '''

            try:
                # load in this dataset
                h5_file = h5py.File(self.h5_path, 'r')

                # dataframe of celltypes and targets
                cellTypes = list(h5_file['rows']["celltypes"][:])
                cellTypes = [i.decode() for i in cellTypes]
                targets = list(h5_file['rows']["targets"][:])
                targets = [i.decode() for i in targets]
                row_df = pd.DataFrame({'cellType': cellTypes,'target':targets})

                # generate 100 records for each chromosome
                PER_CHR_RECORDS=100
                chroms = []
                newStarts = []
                binSize = h5_file['columns']['binSize'][:][0]

                for i in range(1,22): # for all 22 chrs
                    chrom = 'chr' + str(i)
                    for j in range(1,PER_CHR_RECORDS+1):
                        chroms.append(chrom)
                        newStarts.append(j * binSize)

                regions_df = pd.DataFrame({'Chromosome': chroms, 'Start': newStarts})

                # generate random data of 0/1s
                np.random.seed(0)
                rand_data = np.random.randint(2, size=(len(row_df), len(regions_df))).astype('i1')

                EpitomeDataset.save(toy_path,
                        rand_data,
                        row_df,
                        regions_df,
                        binSize,
                        self.assembly,
                        'random_source',
                        valid_chrs = self.valid_chrs,
                        test_chrs = self.test_chrs)

            finally:
                h5_file.close()


    def view(self):
        '''
        Plots a matrix of available targets from available cells.
        '''

        plot_assay_heatmap(self.matrix, self.cellmap, self.targetmap)
