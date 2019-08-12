#############################################################################################
# Downloads DeFCoM data and creates bgzip files and tabix files so it can be used with defcom
#############################################################################################

# Imports
import subprocess
import argparse
import os
import os.path
import pandas as pd

"""
    Takes in a dataframe of positive sites, a dataframe of negative sites and the column names and outputs
    a new dataframe with a mixture of positive and negative sites. It uses all positive sites in the new dataframe
    and random samples the negative sites to have an equal balance of 50/50 for pos and neg.
    
    :param pos_sites_df: Dataframe of positive sites
    :param neg_sites_df: Dataframe of negative sites
    :param colNames: list of column names so it can sort based on chrom_name and chrom_start

    :returns: A dataframe of combined positive and negative sites.
"""
def combine_neg_and_pos(pos_sites_df, neg_sites_df, colNames):
    if(len(pos_sites_df) < len(neg_sites_df)): # let's first make sure it's possible
        neg_sites_df = neg_sites_df.sample(len(pos_sites_df), replace=False)
        neg_sites_df = neg_sites_df.sort_values([colNames[0], colNames[1]])

    all_sites_df = pos_sites_df.append(neg_sites_df)
    all_sites_df = all_sites_df.sort_values([colNames[0], colNames[1]])
    
    return all_sites_df

# Parser for user specific locations
parser = argparse.ArgumentParser(description='Downloads neccessary data for Defcom.')
parser.add_argument('--output_path', help='Path to save data to (creates a new directory here)', required=True)

output_path = parser.parse_args().output_path

if not os.path.exists(output_path):
    os.mkdir(output_path)
    print("{} does not exist. Creating directory. ".format(output_path))

# list of files to download
files_to_download = [
    'http://html-large-files-dept-fureylab.cloudapps.unc.edu/fureylabfiles/defcom/active_gm12878.tar.gz',
    'http://html-large-files-dept-fureylab.cloudapps.unc.edu/fureylabfiles/defcom/inactive_gm12878.tar.gz',
    'http://html-large-files-dept-fureylab.cloudapps.unc.edu/fureylabfiles/defcom/active_k562.tar.gz',
    'http://html-large-files-dept-fureylab.cloudapps.unc.edu/fureylabfiles/defcom/inactive_k562.tar.gz',
    'http://html-large-files-dept-fureylab.cloudapps.unc.edu/fureylabfiles/defcom/active_h1hesc.tar.gz',
    'http://html-large-files-dept-fureylab.cloudapps.unc.edu/fureylabfiles/defcom/inactive_h1hesc.tar.gz',
    'http://html-large-files-dept-fureylab.cloudapps.unc.edu/fureylabfiles/defcom/active_hepg2.tar.gz',
    'http://html-large-files-dept-fureylab.cloudapps.unc.edu/fureylabfiles/defcom/inactive_hepg2.tar.gz'
    ]

# create list of wget calls
wget_calls = ['wget -P {} {}'.format(output_path, download_path) for download_path in files_to_download]

print('Downloading defcom data...')

for call in wget_calls:
    try:
        response        = subprocess.check_output(call, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

print('Done downloading defcom data')

# uncompress everything
files_to_uncompress = [
    'active_gm12878.tar.gz',
    'inactive_gm12878.tar.gz',
    'active_k562.tar.gz',
    'inactive_k562.tar.gz',
    'active_h1hesc.tar.gz',
    'inactive_h1hesc.tar.gz',
    'active_hepg2.tar.gz',
    'inactive_hepg2.tar.gz',
]
uncompress_calls = ['tar -xvzf {} -C {}'.format(os.path.join(output_path,uncompress_path), output_path) for uncompress_path in files_to_uncompress]

print('Uncompressing defcom data...')

for call in uncompress_calls:
    remove_call = 'rm {}'.format(call.split()[2])
    try:
        response = subprocess.check_output(call, shell=True, stderr=subprocess.STDOUT)
        remove_response = subprocess.check_output(remove_call, shell=True, stderr =subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

print('Done uncompressing defcom data...')
        
# list folders in a directory
dir_list = [os.path.join(output_path,folder) for folder in os.listdir(output_path) if os.path.isdir(os.path.join(output_path,folder))]
for data_folder in dir_list:
    move_call = 'mv {}/* {}'.format(data_folder, output_path)
    remove_call = 'rm -rf {}'.format(data_folder)
    try:
        subprocess.check_output(move_call, shell=True, stderr = subprocess.STDOUT)
        subprocess.check_output(remove_call, shell=True, stderr = subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

print('Done temporarily moving defcom data')
print('Now splitting up the data now and creating .gz and .gz.tbi files')

pos_file_list = []

for dir_path, folders, files in os.walk(output_path):
    file_list = [ os.path.join(dir_path, pos_file) for pos_file in files if 'pos' in pos_file and '.tar.gz' not in pos_file]
    pos_file_list.extend(file_list)

# then let's iterate through all the files
colNames = ['chrom_name', 'chrom_start', 'chrom_end', 'strand', 'dunno', 'star', 'score']

# iterate through all of the possible combinations of cell type and tf
for pos_file in pos_file_list:
    # then let's split up the dataframe into three seperate dataframes for pos, neg, and all
    neg_file = pos_file.replace('pos','neg')
    all_file = pos_file.replace('pos','all')
    
    # get the dataframes
    pos_sites_df = pd.read_csv(pos_file, sep='\t', index_col=False, names = colNames)
    neg_sites_df = pd.read_csv(neg_file, sep='\t', index_col=False, names = colNames)
    # let's random sample the inactive sites to get the same number as active binding sites
    
    # now we need to split each of these dataframes up
    pos_sites_train_df =  pos_sites_df.loc[ (pos_sites_df['chrom_name'] != 'chr7') & (pos_sites_df['chrom_name'] != 'chr8') & (pos_sites_df['chrom_name'] != 'chr9') ]
    pos_sites_valid_df =  pos_sites_df.loc[ (pos_sites_df['chrom_name'] == 'chr7')]
    pos_sites_test_df =   pos_sites_df.loc[ (pos_sites_df['chrom_name'] == 'chr8') | (pos_sites_df['chrom_name'] == 'chr9') ]
    neg_sites_train_df =  neg_sites_df.loc[ (neg_sites_df['chrom_name'] != 'chr7') & (neg_sites_df['chrom_name'] != 'chr8') & (neg_sites_df['chrom_name'] != 'chr9') ]
    neg_sites_valid_df =  neg_sites_df.loc[ (neg_sites_df['chrom_name'] == 'chr7')]
    neg_sites_test_df =   neg_sites_df.loc[ (neg_sites_df['chrom_name'] == 'chr8') | (neg_sites_df['chrom_name'] == 'chr9') ]
    
    # create the dataframe for all of the sites
    all_sites_train_df = combine_neg_and_pos(pos_sites_train_df, neg_sites_train_df, colNames)
    all_sites_valid_df = combine_neg_and_pos(pos_sites_valid_df, neg_sites_valid_df, colNames)
    all_sites_test_df  = combine_neg_and_pos(pos_sites_test_df, neg_sites_test_df, colNames)        
    
    # create the path/names for the output files
    pos_train_file_out = pos_file.replace('pos','pos_train')
    pos_valid_file_out = pos_file.replace('pos','pos_valid')
    pos_test_file_out =  pos_file.replace('pos','pos_test')
    neg_train_file_out = neg_file.replace('neg','neg_train')
    neg_valid_file_out = neg_file.replace('neg','neg_valid')
    neg_test_file_out =  neg_file.replace('neg','neg_test')
    all_train_file_out = neg_file.replace('neg','all_train')
    all_valid_file_out = neg_file.replace('neg','all_valid')
    all_test_file_out =  neg_file.replace('neg','all_test')

    # add all the files to a list so we can iterate over it easily
    out_file_list = [
        (pos_sites_train_df, pos_train_file_out),
        (pos_sites_valid_df, pos_valid_file_out),
        (pos_sites_test_df, pos_test_file_out),
        (neg_sites_train_df, neg_train_file_out),
        (neg_sites_valid_df, neg_valid_file_out),
        (neg_sites_test_df, neg_test_file_out),
        (all_sites_train_df, all_train_file_out),
        (all_sites_valid_df, all_valid_file_out),
        (all_sites_test_df, all_test_file_out)
    ]
    
    # now for each of these files let's gunzip them and create a tabix file for them\
    # so defcom and epitome can use them
    for sites_df, output_file_path in out_file_list:
        sites_df.to_csv(output_file_path, sep='\t', header=False, index=False) # should i output them as csvs orrrrr        
        bgzip_call = 'bgzip -c {} > {}.gz'.format(output_file_path, output_file_path)
        tabix_call = 'tabix -p bed {}.gz'.format(output_file_path)
        try:
            bgzip_response = subprocess.check_output(bgzip_call,  shell=True, stderr =subprocess.STDOUT)
            tabix_response = subprocess.check_output(tabix_call, shell=True, stderr =subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

print('Done getting defcom data!')
