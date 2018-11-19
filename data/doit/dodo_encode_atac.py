########################################################################
# ENCODE Config file for SeqTools pipeline.
# Alyssa Morrow
# 11/15/2018
########################################################################

""" This file is a Python DoIt script which creates the analysis for the
the ENCODE A549 ATAC-seq data 
(https://www.encodeproject.org/search/?type=Experiment&assay_title=ATAC-seq&biosample_term_name=A549). It uses the SeqTools project (Yosef Lab), 
which can be found at https://github.com/YosefLab/SeqTools. """


###############################################################################
# Imports
###############################################################################
import seqtools_settings as seqtools_settings



import sys

import os
import itertools
import shutil
import glob

sys.path.append('/home/eecs/akmorrow/SeqTools')
sys.path.append('/home/eecs/akmorrow/yosef/format_seq_qc_output')

# Add a path to wherever you installed seqtools
import SeqTools as st
import SeqTools.dodo_functions as df
import SeqTools.common_pipeline_functions.alignment_and_qc as st_alnQC
import SeqTools.common_pipeline_functions.peak_calling_and_merging as st_peaks
import SeqTools.common_pipeline_functions.shiny as st_shiny

import subprocess as sp

from doit import create_after


###############################################################################
# Define globals and constants.
###############################################################################

dirProjectFolder = seqtools_settings.dirProjectFolder
strConfigFile = seqtools_settings.strConfigFile

dictConfig = seqtools_settings.dictConfig

dirProjectOut = os.path.join(dirProjectFolder, "out")
strOutput = os.path.join(dirProjectFolder, "ConfigFile_Tutorial_Processed.tab")

dirShinyOut = os.path.join(dirProjectOut, "shiny")
bedUniverse = os.path.join(dirProjectOut, "PeakUniverse.bed")
bedUniverseWithCuts = os.path.join(dirProjectOut,"PeakUniverseWithCuts.bed")


########################################################################
# Main Doit Script
########################################################################

print("Running the Project DoIt Script....")

# Load the table of objects

aAlignedSeq = []
dict_sdSamples= {}
aSeqData = st.LoadSeqDataTable(strConfigFile)
for sdSample in aSeqData:
    df.SetTargets(sdSample,dirOut=dirProjectOut)
    dict_sdSamples[sdSample.strSampleName] = sdSample


# print sample information
for sdSample in dict_sdSamples.values():
    if sdSample.strExperimentType == "atac":
        print("Sample: " + sdSample.strSampleName)
        print("Raw Bam File: " + sdSample.bamRawData)
        print("Aligned Output File: " + sdSample.bamReadyForAnalysis + "\n\n")
        print("QC folder: " + sdSample.dirQC + "\n\n")
        print("Cut Sites: " + sdSample.bedCuts + "\n\n")
        
        
# def task_AlignData_RunQC():
#     """ For each sample, align the raw bam file and perform QC.
#     ATAC-Seq and ChIP-Seq are handled differently, see above.
#     Add the modified SeqData object to aAlignedSeq. """
    
#     for sdSample in dict_sdSamples.values():
#         if sdSample.strExperimentType in ["atac"]:
            
#             dirQCFmt = os.path.join(sdSample.dirQC, "clean_qc_out")
#             st.check_create_dir(dirQCFmt)
#             tabFlagstat = os.path.join(dirQCFmt,"flagstatOnRAWbam_stats.tab")

#         # do not perform if fastqs are given
#         if (sdSample.bamRawData != ""): 
#             yield {'name':"FlagStatOnRawBam:" + sdSample.strSampleName,
#                     'actions':[ ( df.FlagstatRawBAM,[sdSample],{}) ], 
#                     'targets':[tabFlagstat],
#                     'file_dep':[sdSample.bamRawData]}
            
#             bBamOutExists = os.path.isfile(sdSample.bamReadyForAnalysis)
#             print("starting ATACQC", sdSample.bamReadyForAnalysis, sdSample, sdSample.bamReadyForAnalysis, bBamOutExists, sdSample.bamRawData)
#             if not bBamOutExists:
#                 print("adding basic ATACQC_version2")
#                 yield {'name':"BasicATACQC_version2:" + sdSample.strSampleName,
#                     'actions':[ ( df.BasicAlignmentAndQC_ATAC_v2,[sdSample],{}) ], 
#                     'targets':[sdSample.bamReadyForAnalysis],'uptodate':[bBamOutExists],
#                     'file_dep':[sdSample.fastq1RawData]}

#         aAlignedSeq.append(sdSample)



def task_Call_Peaks_Create_Universe():

   # Set QVals for each experiment type. The ATAC qvalue is ridiculous to
   # make sure we get peaks. Don't use one this high for real analysis!
   dictPeakQVal = {'atac':0.001}
   abedAllPeaks= []

   # Loop through samples, call peaks with settings based on experiment type.
   for sdSample in dict_sdSamples.values():


       # Name the peak files. These names are long, but they're helpful when you have 100 samples and want to write a
       # unix "glob" to get everything with a certain qvalue, or cat all your files and analyze them in R.
       strPeakName = "__".join([sdSample.strExperimentType,   sdSample.strSampleName,  str(dictPeakQVal[sdSample.strExperimentType])  ]) # Example: chip__Sample1__0.05
       bedPeaks = "/".join([dirProjectOut,"peaks",sdSample.strExperimentType, strPeakName, strPeakName+"_peaks.bed" ]) 


       print("strPeakName", strPeakName, "tarrget bedPeaks", bedPeaks, "bedCuts", sdSample.bedCuts)



       #Produce file of ATAC-Seq cut sites.
       yield {'name':"GetCutSites:" + sdSample.strSampleName,
                'actions':[ ( st_alnQC.GetTn5CutSites,[], {'sdSample':sdSample,'dictConfig':dictConfig}),],
                'targets':[sdSample.bedCuts],
                'file_dep':[sdSample.bamReadyForAnalysis]}   

       if sdSample.strExperimentType in ["atac"]:
           yield {'name':"Call_ATAC_Peak:" + sdSample.strSampleName,
               'actions':[ ( st_peaks.Call_ATACPeak_Fragments,[],
                            # Function arguments.
                            {'sdSample':sdSample,
                            'cmd_MACS2':"/usr/local/bin/macs2",'dictConfig':dictConfig,
                            'dQVal':dictPeakQVal[sdSample.strExperimentType],
                            'bedPeaks':bedPeaks,
                            'bBroad':False,'astrMoreArgs':["--keep-dup","all"],'bRunOverlaps':True})  ],
               'targets': [bedPeaks],
               'file_dep':[sdSample.bamReadyForAnalysis]}
           abedAllPeaks.append(bedPeaks)


   # Merge the peaks to get Universe
   yield {'name':"Merge All The Peaks",
       'actions':[ ( st_peaks.MergePeaks,[],
                    # Function arguments.
                    {'bedMerged':bedUniverse,
                    'abedPeaks':abedAllPeaks})  ],
       'targets':[bedUniverse],
       'file_dep':abedAllPeaks}


   # Get ATAC-Seq Cuts in Peaks
   yield {'name':"Get ATAC-Cuts in Peaks",
       'actions':[ (st_peaks.GetCutSitesInPeaks,[],{'bedUniverse':bedUniverse,
                                   'tabCuts':bedUniverseWithCuts,
                                   'asdSample':[sdSample for sdSample in dict_sdSamples.values() if sdSample.strExperimentType =="atac"],
                                                })
     ],
       'targets':[bedUniverseWithCuts],
       'file_dep':[bedUniverse]+ [sdSample.bedCuts for sdSample in dict_sdSamples.values() if sdSample.strExperimentType =="atac"],
       'uptodate':[False]}
