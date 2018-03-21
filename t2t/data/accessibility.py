# This file contains all code for parsing accessibility data (DNase or ATAC-seq)
# for tensor2tensor

import numpy as np
import pybedtools

def get_accessibility_vector(chr_, start, stop, accessibility_df):
	'''
	Returns an cut vector of length start - stop
	:param chr_ chr to access
	:param start start of region to process
	:param stop end of region to process
	:param accessibility_path path to bed file of DNase or ATAC seq data
	as processed by seqOutBias. Example data in c66:/data/epitome/accessibility
	'''

	filtered_bed_df = accessibility_df[(accessibility_df['chr'] == chr_) & (accessibility_df['start'] < stop)& (accessibility_df['stop'] > start)]

	vector = np.zeros(length)

	# TODO inefficient
	for i, row in filtered_bed_df.iterrows():
		vector[row['start'] - start] = row['value']

	return vector

def get_accessibility_vector_pybed(i, accessibility_data):
	'''
	Returns the cut vector corresponding to ith row
	@param i index of the current row
	@param accessibility_data the pybedtool that reads the peak data
	'''
	fields = accessibility_data[i].fields
	start, positions, accessibility = int(fields[1]), fields[9], fields[11]
	if positions == '.':
		return np.zeros(1000)
	start = start - 400
	positions = np.array(list(map(int, positions.split(','))))
	accessibility = list(map(float, accessibility.split(',')))
	vector = np.zeros(1000)
	positions = positions - start
	vector[positions] = accessibility
	return vector