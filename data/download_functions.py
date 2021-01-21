# Functions for downloading data.
#



############################## Imports ####################################

import logging
import pandas as pd
import numpy as np
import pyranges as pr
import os
import sys
import h5py
import subprocess
from epitome.dataset import *
import urllib
import re

############################## Constants ######################################
# for accessing metadata for both ChIPAtlas and ENCODE
COL_CELLTYPE = "Cell type"
COL_ANTIGEN = "Antigen"
COL_CLASS = "Antigen class"
COL_ID = "Experimental ID"

##################################### LOG INFO ################################
def set_logger():
    """
    Set up logger

    Args:
        :param name: logger name
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create console handler with a low log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    return logger

logger = set_logger()

def count_lines(file_path):
    """
    Counts number of lines in a file

    Args:
        :param file_path: path to file

    :return int: number of lines in file
    """
    count = 0
    for line in open(file_path).readlines(  ): count += 1
    return count

def lojs_overlap(feature_files, compare_pr):
    """
    Function to run left outer join in features to all_regions_file

    Args:
            :param feature_files: list of paths to file to run intersection with all_regions_file
            :param compare_pr: pyranges object containing all regions of interest

    :return arr: array same size as the number of genomic regions in all_regions_file
    """

    if len(feature_files) == 0:
        logger.warn("WARN: lojs_overlap failed for all files %s with 0 lines" % ','.join(feature_files))
        return np.zeros(len(compare_pr))

    #### Number of files that must share a consensus ####
    if len(feature_files)<=2:
        n = 1 # if there are 1-2 files just include all
    elif len(feature_files) >=3 and len(feature_files) <= 7:
        n=2
    else:
        n = int(len(feature_files)/4) # in 25% of files

    # Very slow: concatenate all bed files and only take regions with n overlap
    group_pr = pr.concat([pr.read_bed(i).merge(slack=20) for i in feature_files])
    group_pr = group_pr.merge(slack=20, count=True).df
    group_pr = group_pr[group_pr['Count']>=n]

    # Remove count column and save to bed file
    group_pr.drop('Count', inplace=True, axis=1)

    pr1 = pr.PyRanges(group_pr)

    intersected = compare_pr.count_overlaps(pr1)
    arr = intersected.df['NumberOverlaps'].values
    arr[arr>0] = 1
    return arr


def download_ENCODE_url(f, bed_download_path, bigBedToBed, tries = 0):


    if tries == 2:
        raise Exception("File accession %s from URL %s failed for download 3 times. Exiting 1..." % (f[COL_ID], f["File download URL"]))

    path = f["File download URL"]
    id = f[COL_ID]

    ext = path.split(".")[-1]
    if (ext == "gz" and path.split(".")[-2]  == 'bed'):
        ext = "bed.gz"

    file_basename = os.path.basename(path).split('.')[0]
    outname_bb = os.path.join(bed_download_path, "%s.%s" % (file_basename, ext))
    outname_bed = os.path.join(bed_download_path, "%s.%s" % (file_basename, 'bed'))

    # make sure file does not exist before downloading
    try:
        if not os.path.exists(outname_bed):

            # download if not yet downloaded
            if not os.path.exists(outname_bb):
                if sys.version_info[0] < 3:
                    # python 2
                    urllib.urlretrieve(path, filename=outname_bb)
                else:
                    # python 3
                    urllib.request.urlretrieve(path, filename=outname_bb)

            if (ext == "bed.gz"):
                subprocess.check_call(["gunzip","-f",outname_bb])
            elif (ext == "bigBed"):
                subprocess.check_call([bigBedToBed, outname_bb, outname_bed])
                os.remove(outname_bb)

        return outname_bed
    except:
        # increment tries by one and re-try download
        return download_ENCODE_url(f, tries + 1)

def download_CHIPAtlas_url(f, bed_download_path, tries = 0):
    '''
    Downloads a file from filtered_files dataframe row

    Returns path to downloaded file
    '''

    path = f["Peak-call (BED) (q < 1E-05)"]
    id_ = f[COL_ID]
    file_basename = os.path.basename(path)

    if tries == 2:
        raise Exception("File accession %s from URL %s failed for download 3 times. Exiting 1..." % (id_, path))

    outname_bed = os.path.join(bed_download_path, file_basename)

    # make sure file does not exist before downloading
    try:
        if not os.path.exists(outname_bed):

            logger.warning("Trying to download %s for the %ith time..." % (path, tries))

            if sys.version_info[0] < 3:
                # python 2
                urllib.urlretrieve(path, filename=outname_bed)
            else:
                # python 3
                urllib.request.urlretrieve(path, filename=outname_bed)
    except urllib.error.HTTPError as e:
        logger.warning("Could not download %s with error %s" % (path,e))
        return None
    except:
        # increment tries by one and re-try download
        return download_CHIPAtlas_url(f, tries + 1)

    return outname_bed

def get_metadata_groups(metadata_file, assembly, min_chip_per_cell = 1 ,min_cells_per_chip = 3):
    """
    Gets metadata from file and groups by cell/target. Can filter either ChIP-Atlas or ENCODE file.

    Args:
        :param metadata_file: local path to unzipped csv file
        :param assembly: genome assembly
        :param min_chip_per_cell: min chip experiments required for a cell type to be considered. Default = 1
        :param min_cells_per_chip: min cells required for each chip experiment. Default = 3

    :return pd: pandas grouped dataframe of experiments, grouped by (cell type, ChIP-seq target)

    """

    if "chip_atlas" in metadata_file:
        return get_metadata(metadata_file,
            assembly,
            "CHIPATLAS",
            min_chip_per_cell = min_chip_per_cell,
            min_cells_per_chip = min_cells_per_chip)
    elif "encode" in metadata_file:
        return get_metadata(metadata_file,
            assembly,
            "ENCODE",
            min_chip_per_cell = min_chip_per_cell,
            min_cells_per_chip = min_cells_per_chip)
    else:
        raise Exception("Unknown metadata file %s " % metadata_file)

def get_metadata(metadata_file,
                           assembly,
                           data_source,
                           min_chip_per_cell = 1,
                           min_cells_per_chip = 3):
    """
    Gets ChIP-Atlas/ENCODE metadata for a specific assembly. Metadata file can be
    downloaded from ftp://ftp.biosciencedbc.jp/archive/chip-atlas/LATEST/chip_atlas_experiment_list.zip
    or http://www.encodeproject.org/metadata/type%3DExperiment%26assay_title%3DTF%2BChIP-seq%26assay_title%3DHistone%2BChIP-seq%26assay_title%3DDNase-seq%26assay_title%3DATAC-seq%26assembly%3Dhg19%26files.file_type%3DbigBed%2BnarrowPeak/metadata.tsv

    Args:
        :param metadata_file: local path to unzipped csv file
        :param assembly: genome assembly
        :param data_source: what is the data source. Either "CHIPATLAS" or "ENCODE"
        :param min_chip_per_cell: min chip experiments required for a cell type to be considered. Default = 1
        :param min_cells_per_chip: min cells required for each chip experiment. Default = 3

    :return pd: pandas grouped dataframe of experiments, grouped by (cell type, ChIP-seq target)
    """

    assert data_source in ["CHIPATLAS","ENCODE"],  "Unknown datasource %s" % data_source

    if metadata_file.endswith('.csv'):
        ######### CHIP-Atlas specific code
        files = pd.read_csv(metadata_file, engine='python') # needed for decoding

        # rename antigen class cols to match encode: ATAC-seq, DNase-seq, Histone ChIP-seq, TF ChIP-seq
        files[COL_CLASS].replace('TFs and others', 'TF ChIP-seq', inplace=True)
        files[COL_CLASS].replace('Histone', 'Histone ChIP-seq', inplace=True)
        # make DNase-seq consistent with ENCODE
        files[COL_ANTIGEN].replace('DNase-Seq', 'DNase-seq', inplace=True)


    elif metadata_file.endswith('.tsv'):
        ######## ENCODE specific filtering code ###################
        files = pd.read_csv(metadata_file, sep="\t")

        # rename everything to match CHIP-Atlas
        files.rename(columns={"Assay": COL_CLASS,
                              "Biosample term name": COL_CELLTYPE,
                              "Experiment target": COL_ANTIGEN,
                              "File accession": COL_ID}, inplace=True)

        # remove -human and -mouse suffix from Antigen column
        d = list(files[ COL_ANTIGEN].str.split('-'))
        genomes = list(set([i[-1] if type(i)== list else None for i in d]))
        genomes = [i for i in genomes if i is not None]
        for i in genomes:
            files[COL_ANTIGEN] = files[COL_ANTIGEN].str.rstrip('-' + i)

        # consistently name Antigen column for DNase-seq
        files.loc[files[COL_CLASS] == 'DNase-seq', COL_ANTIGEN ] = 'DNase-seq'
        files.loc[files[COL_CLASS] == 'ATAC-seq', COL_ANTIGEN ] = 'ATAC-seq'




    else:
        raise Exception("Cannot load %s. File type not recognized." % metadata_file)

    # assembly column is either 'Assembly' or 'File assembly'
    assembly_column = files.filter(regex=re.compile('Assembly', re.IGNORECASE)).columns[0]
    assert assembly in list(set(files[assembly_column])), "Assembly %s is not in column %s" % (assembly, ','.join(list(set(files[assembly_column]))))

    antigen_classes = ['ATAC-seq', 'DNase-seq', 'Histone ChIP-seq', 'TF ChIP-seq']

    assembly_files = files[(files[assembly_column] == assembly) &
                                   (files[COL_CLASS].isin(antigen_classes))]

    if data_source == "ENCODE":
        # extra filtering step for ENCODE: remove assays with errors and treatments
        assembly_files = assembly_files[(assembly_files["Audit ERROR"].isnull()) &
                          (assembly_files["Biosample treatments"].isnull())]

        # Get unique by Antigen, Cell type, file type

        rm_dups = assembly_files[[COL_ANTIGEN, COL_CELLTYPE,"Output type"]].drop_duplicates()
        filtered_files = assembly_files.loc[rm_dups.index]
    else:
        # Get unique by Antigen, Cell type

        rm_dups = assembly_files[[COL_ANTIGEN, COL_CELLTYPE]].drop_duplicates()
        filtered_files = assembly_files.loc[rm_dups.index]

    # get unique dnase experiments
    filtered_dnase = filtered_files[((filtered_files[COL_CLASS] == "DNase-seq"))]
    if data_source == "ENCODE":
        # ENCODE: filter out other file types
        filtered_dnase = filtered_dnase[(filtered_dnase["Output type"] == "peaks")]

    chip_files = filtered_files[filtered_files[COL_CLASS].str.contains("ChIP-seq")]
    if data_source == "ENCODE":
        # ENCODE: filter out other file types
        chip_files = chip_files[(chip_files["Output type"] == "replicated peaks") |
                                      (chip_files["Output type"] == "optimal IDR thresholded peaks")]

    # only want ChIP-seq from cell lines that have DNase
    filtered_chip = chip_files[(chip_files[COL_CELLTYPE].isin(filtered_dnase[COL_CELLTYPE]))]

    # only want assays that are shared between more than 3 cells
    filtered_chip = filtered_chip.groupby(COL_ANTIGEN).filter(lambda x: len(x) >= min_cells_per_chip)

    # only want cells that have more than min_chip_per_cell epigenetic marks
    filtered_chip = filtered_chip.groupby(COL_CELLTYPE).filter(lambda x: len(x) >= min_chip_per_cell)

    # only filter if use requires at least one chip experiment for a cell type.
    if min_chip_per_cell > 0:
        # only want DNase that has chip.
        filtered_dnase = filtered_dnase[(filtered_dnase[COL_CELLTYPE].isin(filtered_chip[COL_CELLTYPE]))]

    # get ATAC-seq data: ENCODE only
    filtered_atac = filtered_files[filtered_files[COL_CLASS].str.contains("ATAC-seq")]
    if data_source == "ENCODE":
        filtered_atac = filtered_atac[(filtered_atac["Output type"] == "IDR thresholded peaks")]
    filtered_atac = filtered_atac[(filtered_atac[COL_CELLTYPE].isin(filtered_dnase[COL_CELLTYPE]))]

    # combine dataframes
    filtered_files = filtered_dnase.append(filtered_chip).append(filtered_atac)
    filtered_files.reset_index(inplace = True)

    # go back to original dataset and select all files that have the selected antigen/cell types
    replicate_groups = assembly_files[(assembly_files[COL_ANTIGEN].isin(filtered_files[COL_ANTIGEN])) &
                                      (assembly_files[COL_CELLTYPE].isin(filtered_files[COL_CELLTYPE]))]

    # read in annotated Antigens
    this_dir = os.path.dirname(os.path.abspath(__file__))
    TF_categories = pd.read_csv(os.path.join(this_dir,'ChIP_target_types.csv'),sep=',')
    TF_categories.replace({'DNase': 'DNase-seq'}, inplace=True)
    TF_categories.replace({'ATAC': 'ATAC-seq'}, inplace=True)


    # sanity check that all antigens are accounted for in TF_categories
    missing = [i for i in set(replicate_groups[COL_ANTIGEN]) if i not in list(TF_categories['Name'])]
    if len(missing)>0:
        logging.error("Missing antigens:")
        logging.error(','.join(missing))

    assert len(missing) == 0, "Missing %i antigens" % len(missing)

    # Filter out ChIP-seq not in TFs, accessibility, histones, etc. We lose about 1100 rows
    filtered_names = TF_categories[TF_categories['Group'].isin(['TF','chromatin accessibility','chromatin modifier','histone',
     'histone modification'])]

    replicate_groups = replicate_groups[replicate_groups[COL_ANTIGEN].isin(filtered_names['Name'])]
    replicate_groups.reset_index(inplace = True)

    logger.info("Processing %i antigens and %i experiments" % (len(set(replicate_groups[COL_ANTIGEN])), len(replicate_groups)))

    # we use this later on to determine whether the dataframe is ENCODE or CHIPATLAS
    replicate_groups['Source'] = data_source

    # group experiments together
    return replicate_groups.groupby([COL_ANTIGEN, COL_CELLTYPE])


def save_epitome_dataset(download_dir,
                        output_path,
                        matrix_path,
                        all_regions_file_unfiltered,
                        row_df,
                        assembly,
                        source):
    """
    Saves epitome labels as numpy arrays, filtering out training data for 0 vectors.

    Args:
        :param download_dir: Directory containing train.h5, all.pos.bed file and feature_name file.
        :param output_path: new output path. saves as numpy files.
        :param matrix_path: path to unfiltered, sparse h5 matrix of 0/1s
        :param all_regions_file_unfiltered: path to bed file containing genome regions, matching
            the number of columns in matrix_path
        :param row_df: dataframe containing columns (cellType, target) matching number of rows in matrix_path.
        :param assembly: genome assembly
        :param source: string source

    """

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        logger.info("%s Created " % output_path)

    # load in all data into RAM: much faster for indexing
    tmp = h5py.File(matrix_path, "r")
    h5_data = tmp['data'][:,:]
    tmp.close()
    logger.info("loaded data..")

    # get non-zero indices
    nonzero_indices = np.where(np.sum(h5_data, axis=0) > 0)[0]

    ## get data from columns where sum > 0
    nonzero_data = h5_data[:,nonzero_indices]
    logger.info("number of new indices is %i" % nonzero_indices.shape[0])

    # filter and re-save all_regions_file
    regions_df = pd.read_csv(all_regions_file_unfiltered, sep='\t', header=None).loc[nonzero_indices]
    regions_df.rename(columns={0:'Chromosome',1:'Start'}, inplace=True)

    EpitomeDataset.save(output_path,
        nonzero_data,
        row_df,
        regions_df,
        200,
        assembly,
        source,
        valid_chrs = ['chr7'],
        test_chrs = ['chr8','chr9'])



def window_genome(all_regions_file_unfiltered,
                download_path,
                assembly):
    """
    Window genome into 200bp regions. Remove nonautosomal chromosomes.

    Args:
        :param all_regions_file_unfiltered: ungzipped bed file to save regions to
        :param download_path: where to download files to
        :param assembly: genome assembly

    Returns:
        number of regions in all_regions_file_unfiltered bed file

    """
    # get chrom sizes file and make windows for genome
    if not os.path.exists(all_regions_file_unfiltered):
        tmpFile = all_regions_file_unfiltered + ".tmp"
        chrom_sizes_file = os.path.join(download_path, "%s.chrom.sizes" % assembly)

        # download file
        if not os.path.exists(chrom_sizes_file):
            if assembly == 'GRCh38':
                # special case => hg38 in UCSC
                subprocess.check_call(["wget", "-O", chrom_sizes_file, "-np", "-r", "-nd", "https://genome.ucsc.edu/goldenPath/help/%s.chrom.sizes" % 'hg38'])
            else:
                subprocess.check_call(["wget", "-O", chrom_sizes_file, "-np", "-r", "-nd", "https://genome.ucsc.edu/goldenPath/help/%s.chrom.sizes" % assembly])

        # window genome into 200bp regions
        if not os.path.exists(tmpFile):
            stdout = open(tmpFile,"wb")
            subprocess.call(["bedtools", "makewindows", "-g", chrom_sizes_file, "-w", "200"],stdout=stdout)
            stdout.close()

        # filter out chrM, _random and _cl chrs
        stdout = open(all_regions_file_unfiltered,"wb")
        subprocess.check_call(["grep", "-vE", "_|chrM|chrM|chrX|chrY", tmpFile], stdout = stdout)
        stdout.close()

        # remove tmp file
        os.remove(tmpFile)
