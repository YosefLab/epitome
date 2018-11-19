# Alyssa Morrow 11/19/2018
# This script was run on yoseflab s124 machine
# requirements: idr 
# Files were download from ENCODE website and processed with the doit scripts 

idr --samples /data/yosef2/Epitome/A549/out/peaks/atac/atac__ENCFF022FVS__0.001/atac__ENCFF022FVS__0.001_peaks.narrowPeak /data/yosef2/Epitome/A549/out/peaks/atac/atac__ENCFF224EAV__0.001/atac__ENCFF224EAV__0.001_peaks.narrowPeak --input-file-type narrowPeak --output-file /data/yosef2/Epitome/A549/out/peaks/atac/idr_out/idr.idr0.1.unthresholded-peaks.txt --rank p.value --soft-idr-threshold 0.1 --plot --use-best-multisummit-IDR --log-output-file /data/yosef2/Epitome/A549/out/peaks/atac/idr_out/idr.idr0.1.log
