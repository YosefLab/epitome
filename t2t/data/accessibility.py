# This file contains all code for parsing accessibility data (DNase or ATAC-seq)
# for tensor2tensor

import numpy as np

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
