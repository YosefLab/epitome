
###############################################################################
# Config File and Output
###############################################################################
import os

# This will be your default output folder
dirProjectFolder = "/data/yosef2/Epitome/A549"

# This is your config file.
strConfigFile = os.path.join(dirProjectFolder,"Config_A549_ATAC.tab") # contains file locations and information
# where data is located on s124


###############################################################################
# Reference files for genomes
###############################################################################



# QCles
dictGenome_bowtie2 = {'mm10':'/data/yosef/index_files/mm10/mm10_bowtie2index/mm10',
                        'hg19':'/data/yosef/index_files/hg19/genome/bowtie2_ind/hg19',
                        'dm3':'/data/yosef/index_files/dm3/dm3_bowtie2index/genome'}
                        
# ENCODE blacklist regions
dictBlacklist = {'mm10':'/data/yosef/index_files/mm10/other_annot/mm10-blacklist.bed',
'hg19':'/data/yosef/index_files/hg19/annotation/hg19-blacklist-kundaje.bed',
'dm3':'/data/yosef/index_files/dm3/other_annot/dm3-blacklist.bed'}

# List of regions highlighted by Buenrosto as containing regions similar to 
# ChrM. Typically filtered out in ATAC-Seq data.
dictBlacklistLikeChrM = {'mm10':'/data/yosef/index_files/mm10/other_annot/JDB_blacklist.mm10.bed',
                    'hg19':'/data/yosef/index_files/hg19/annotation/JDB_blacklist.hg19.bed'}

# Tab-delimited file listing the length of each chromosome       
dictGenomeSizes = {'mm10':'/data/yosef/index_files/mm10/genome/mm10_sizes.tab',
                   'hg19':'/data/yosef/index_files/hg19/genome/hg19.chrom.sizes.txt',
                   #'dm3':'/data/yosef/index_files/dm3/genome/dm3_sizes.tab'
                   }
                   
# Sets appropriate macs2 settings for genome
dictMacs2Genome = {'mm10':'mm','hg19':'hs','dm3':'dm'}

# Default annotation file for overlaps database. NOTE: You will most likely 
# want to create your own!
dictAnnotationFile = {'mm10':'/data/yosef2/BATF_Project/src/annotation_lists/mm10_annotation.tab',
                        'hg19':'/data/yosef2/BATF_Project/src/BedOverlapsDB/hg19_annotation_quicktest.tab'}

# List of transcription start sites.
dictTSS = {'mm10':'/data/yosef/index_files/mm10/other_annot/TSS_RefSeqUpstream1_mm10_uniq_sorted.bed',
           'hg19':'/data/yosef/index_files/hg19/genes/hg19_cutsites_UCSC_upstream1bp_uniq.bed'}


###############################################################################
# Commands
###############################################################################

dictCmd = {
    # Alignment and QC
    'trimmomatic':"/opt/pkg/Trimmomatic-0.32/trimmomatic-0.32.jar",
    'fastqc':"/opt/pkg/fastqc-v0.11.5/fastqc",
    'samtools':"/opt/pkg/samtools-1.3.1/bin/samtools",
    'bowtie2':"/opt/pkg/bowtie2-2.2.9/bowtie2",
    'igvtools':"/home/eecs/akmorrow/Programs/IGVTools/igvtools",
        
    # Picard has many tools, so we just call its folder and a particular name
    #   Example: [dictConfig["dictCmd"]["dirPicard"]+"/SortSam"] to call SortSam.
    'dirPicard':"/opt/pkg/picard-tools-2.5.0/bin/",

    # Other    
    'macs2':"/usr/local/bin/macs2", # Change this to have a version number in the call.
    'bedtools':"/opt/genomics/bin//bedtools",
    'pyOverlaps':"/home/eecs/jimkaminski/BedOverlapsDB/OverlapAnnotationsWithBed.py",    

    # Interpreters
    'python2.7':"/usr/bin/python2.7",
    'python3':"/data/yosef/anaconda/envs/py3/bin/python3.5",
    'Rscript':"/usr/bin/Rscript",
    'java':"/usr/lib/jvm/java-8-oracle/jre/bin/java",
    'perl':"/usr/bin/perl"
}
    
dictConfig = {
    'dictCmd':dictCmd,
    'dictGenome_bowtie2':dictGenome_bowtie2,
    'dictBlacklist':dictBlacklist,
    'dictBlacklistLikeChrM':dictBlacklistLikeChrM,
    'dictGenomeSizes':dictGenomeSizes,
    'dictMacs2Genome':dictMacs2Genome,
    'dictAnnotationFile':dictAnnotationFile,
    'dictTSS':dictTSS
}
