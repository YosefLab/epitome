#############################################################################
# Downloads DeepSEA data for training and extracts labels to be used in model
#############################################################################

# Imports
from epitome.functions import *
import subprocess
import argparse

# Parser for user specific locations
parser = argparse.ArgumentParser(description='Runs Epitome on a directory of chromatin bed files.')
parser.add_argument('--output_path', help='Path to save data to (creates a new directory here)', required=True)

output_path = parser.parse_args().output_path

if not os.path.exists(output_path):
    os.mkdir(output_path)
    print("%s does not exist. Creating directory. " % output_path)

# output paths for saved DeepSEA data
tar_path = os.path.join(output_path, "deepsea_train_bundle.v0.9.tar.gz")
deepsea_path = os.path.join(output_path, "deepsea_train") 

calls = []


# if tar file does not exist, download it
if not os.path.isfile(tar_path):
    wget_call = "wget -P %s http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz" % output_path
    calls.append(wget_call)

# if uncompressed folder does not exist, uncompress it
if not os.path.isdir(deepsea_path):
    tar_call = "tar -C %s -zxvf %s" % (output_path, tar_path)
    calls.append(tar_call)
 
# Run download_deepsea_data.sh to get data

for call in calls:
    call = call.split()

    response = subprocess.run(call, 
                              stderr=subprocess.PIPE)


# Save labels
label_output_path = os.path.join(output_path, "deepsea_labels_train")
print("Saving deepsea labels to %s..." % label_output_path)

save_deepsea_label_data(deepsea_path, label_output_path)

print("Done! Please delete %s if no longer needed." % tar_path)

