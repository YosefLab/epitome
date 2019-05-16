#####################################################################
#
#
#
#####################################################################

# Imports 
from shutil import copyfile
import pybedtools
import os
import argparse

from epitome.functions import *
from epitome.constants import *

# Absolute path of Metrics folder
current_dirname = os.path.dirname(os.path.abspath(__file__)) # in Metrics

feature_path = os.path.join(current_dirname, '../../data/feature_name')

################################ Parse Args ########################

# Parser for user specific locations
parser = argparse.ArgumentParser(description='Save overlaps response results as np matrix of 0/1s')

parser.add_argument('--deepsea_path', help='deepsea_train labels downloaded from DeepSEA (./bin/download_deepsea_data.sh)')
parser.add_argument('--output_path', help='path to save DeepSEA performance np file to')
parser.add_argument('--motif_path', help='path to motif database')
parser.add_argument('--all_pos_file', help='path to DeepSEA positions')

deepsea_labels = parser.parse_args().deepsea_path
output_path = parser.parse_args().output_path
motif_path = parser.parse_args().motif_path
all_pos_file = parser.parse_args().all_pos_file


if (deepsea_labels == None or output_path == None):
    raise ValueError("Invalid inputs for deepsea_path or output_path")

if (not os.path.isdir(output_path)):
    print("%s not found. Creating directory..." % output_path)
    os.mkdir(output_path)
    

if (not os.path.isfile(all_pos_file)):
    raise ValueError("all_pos_file does not exist")
    
    
########################### End Parse Args #####################

# generated from original data by save_deepsea_label_data(deepsea_path) in functions.py
# train_data, valid_data, test_data = load_deepsea_label_data(deepsea_labels)

matrix, cellmap, assaymap = get_assays_from_feature_file(feature_path=feature_path,eligible_assays = None,
                                  eligible_cells = None, min_cells_per_assay = 2, min_assays_per_cell=2)
                                  

######################## Functions ###############################
def read_positions_file(all_pos_file):
    """
    Reads test positions from position file for DeepSEA.
    
    Returns: list of tab separated regions
    """
    fp = open(all_pos_file)
    lines = []
    for i, line in enumerate(fp):
        if i >= _TEST_REGIONS[0] and i <= _TEST_REGIONS[1]:
            # read test line
            lines.append(line)
    fp.close()
    
    return lines

def get_peaks_for_bed_file(bed_positions, motif_dir):
    """
    Runs an overlap scan for all peaks in bed file on a given motif file.
    :param bed_positions: list of tab delimited genomic regions
    :param motif_dir: dir containing path to all motif files
    
    Returns: A vector of 0/1's indicating whether motif was found in that region
    """
    
    motif_files = os.listdir(motif_dir)
    
    # just get motif name. i.e. CTCF_M4433
    motif_names = list(map(lambda x: x.split('_1.02.bed')[0], motif_files))

    # test # by motif matrix
    response = np.zeros([len(bed_positions), len(motif_files)])

    # load in files
    pos = pybedtools.BedTool(bed_positions)
    pos_l = list(enumerate(pos.features()))
    pos_d = dict((pybedtools.Interval(v.chrom, v.start, v.end),k) for k,v in pos_l)

    for idx, motif_file in enumerate(motif_files):
        print(motif_file)
        # load in motifs for this file                 
        motifs = pybedtools.BedTool(os.path.join(motif_dir, motif_file))

        # get peaks in a that have overlap motif in b
        ones = pos.window(motifs, w=0).overlap(cols=[2,3,8,9])
        ones_l = [pybedtools.Interval(x.chrom, x.start, x.end) for x in ones.features()]

        # iterate through all hits and set to 1
        for i in ones_l:
            response[pos_d[i], idx] = 1

    return response, motif_names
    
    
# load in test positions and create pybedtool
bed_positions = read_positions_file(all_pos_file)

response, motif_names = get_peaks_for_bed_file(bed_positions, motif_path)

print(motif_names)

# write out results
np.save(os.path.join(output_path, "responses.npy"), response)
        
with open(os.path.join(output_path, "responses_colnames.txt"), 'w') as f:
    for item in motif_names:
        f.write("%s\n" % item)
    