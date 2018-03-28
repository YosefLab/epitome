# This file contains all code for parsing accessibility data (DNase or ATAC-seq)
# for tensor2tensor

import numpy as np
import tensorflow as tf
import time
import pybedtools
>>>>>>> 698f7ec... continue data processing

def save_merged_bedfile(all_tfs_pos_filepath, accessibility_filepath, joined_suffix):
	'''
	Merges two bed files
	:param all_tfs_pos_filepath file 1 to merge
	:param accessibility_filepath location of file 2
	:param joined_suffix suffix of new joined file
	:return joined Bedtools filename and object
	'''
	# load postions and accessibility bed files
	sequence_bed = pybedtools.BedTool(all_tfs_pos_filepath)
	cuts_bed = pybedtools.BedTool(accessibility_filepath)
    
	# finds all sites in cuts_bed that overlap sequence_bed records
	windowed = sequence_bed.window(cuts_bed, w=400)
    
	# merge overlapping records. first 3 columns are chr, start and end. 4th column is comma delim list of start
	# 5th column is comma delim list of stop
	# 6th column is comma delim list of values
	merged = windowed.merge(c=[8, 9, 11], o=["collapse", "collapse", "collapse"], d=-1)
    
	# left outer join
	data = sequence_bed.intersect(merged, loj=True, wa=True)
    
	# save and return results
	joined_filepath = accessibility_filepath + joined_suffix
	tf.logging.info("Saving joined bed file for accessibility to %s " % (joined_filepath))
	result = data.saveas(joined_filepath)
	return (joined_filepath, result)

def get_accessibility_vector_pybed(fields):
	'''
	Returns the cut vector corresponding to ith row
	:param i index of the current row
	:param accessibility_data the pybedtool that reads the peak data
	'''
	chr_, start, stop, positions, accessibility = fields[0], int(fields[1]), int(fields[1]), fields[9], fields[11]
	if positions == '.':
		return np.transpose(np.matrix(np.zeros(1000)))
	start = start - 400
	positions = np.array(list(map(int, positions.split(','))))
	accessibility = list(map(float, accessibility.split(',')))
	vector = np.zeros(1000)
	positions = positions - start
	vector[positions] = accessibility
	return np.transpose(np.matrix(vector))

