# Author Alyssa Morrow 11/19/2018
# intended to run on s124 machine
# Note: this is not yet working.

# code downloaded from https://github.com/ENCODE-DCC/atac-seq-pipeline
python /data/yosef/users/akmorrow/code/atac-seq-pipeline/src/encode_idr.py \
	/data/yosef2/Epitome/A549/out/peaks/atac/atac__ENCFF022FVS__0.001/atac__ENCFF022FVS__0.001_peaks.narrowPeak \
	/data/yosef2/Epitome/A549/out/peaks/atac/atac__ENCFF224EAV__0.001/atac__ENCFF224EAV__0.001_peaks.narrowPeak \
	/data/yosef2/Epitome/A549/out/PeakUniverse.bed \
        --peak-type narrowPeak \
        --blacklist /data/yosef/index_files/hg19/annotation/hg19-blacklist-kundaje.bed \
	--out-dir /data/yosef2/Epitome/A549/out/peaks/atac/idr_out 
