# Downloads ENCODE DNase data for 12 cell types
# # Requirements
# - IGVTools: (used for indexing bam files)
# -- https://software.broadinstitute.org/software/igv/download
# -- then add as alias to bash (alias igvtools=<path_to_igvtools>

# A549 hg19 https://www.encodeproject.org/files/ENCFF566SPI/
wget https://www.encodeproject.org/files/ENCFF414MBW/@@download/ENCFF414MBW.bam
FILE=ENCFF414MBW_A549.bam
mv ENCFF414MBW.bam $FILE
igvtools index $FILE

# K562 https://www.encodeproject.org/files/ENCFF441RET/
wget https://www.encodeproject.org/files/ENCFF441RET/@@download/ENCFF441RET.bam
FILE=ENCFF441RET_K562.bam
mv ENCFF441RET.bam ENCFF441RET_K562.bam
igvtools index $FILE


# H1-hESC
wget https://www.encodeproject.org/files/ENCFF571SSA/@@download/ENCFF571SSA.bam
FILE=ENCFF571SSA_H1heSC.bam
mv ENCFF571SSA.bam ENCFF571SSA_H1heSC.bam
igvtools index $FILE

# GM12878
wget https://www.encodeproject.org/files/ENCFF775ZJX/@@download/ENCFF775ZJX.bam
FILE=ENCFF775ZJX_GM12878.bam
mv ENCFF775ZJX.bam ENCFF775ZJX_GM12878.bam
igvtools index $FILE

# HepG2
wget https://www.encodeproject.org/files/ENCFF224FMI/@@download/ENCFF224FMI.bam
FILE=ENCFF224FMI_HepG2.bam
mv ENCFF224FMI.bam ENCFF224FMI_HepG2.bam
igvtools index $FILE

# HeLa-S3 
wget https://www.encodeproject.org/files/ENCFF783TMX/@@download/ENCFF783TMX.bam
FILE=ENCFF783TMX_HeLaS3.bam
mv ENCFF783TMX.bam ENCFF783TMX_HeLaS3.bam
igvtools index $FILE

#HUVEC 
wget https://www.encodeproject.org/files/ENCFF757PTA/@@download/ENCFF757PTA.bam
FILE=ENCFF757PTA_HUVEC.bam
mv ENCFF757PTA.bam ENCFF757PTA_HUVEC.bam
igvtools index $FILE

# GM12891 
wget https://www.encodeproject.org/files/ENCFF070BAN/@@download/ENCFF070BAN.bam
FILE=ENCFF070BAN_GM12891.bam
mv ENCFF070BAN.bam ENCFF070BAN_GM12891.bam
igvtools index $FILE

# MCF-7
wget https://www.encodeproject.org/files/ENCFF441RET/@@download/ENCFF441RET.bam
FILE=ENCFF441RET_MCF7.bam
mv ENCFF441RET.bam ENCFF441RET_MCF7.bam
igvtools index $FILE

# GM12892 
wget https://www.encodeproject.org/files/ENCFF260LKE/@@download/ENCFF260LKE.bam
FILE=ENCFF260LKE_GM12892.bam
mv ENCFF260LKE.bam ENCFF260LKE_GM12892.bam
igvtools index $FILE

#HCT-116
wget https://www.encodeproject.org/files/ENCFF291HHS/@@download/ENCFF291HHS.bam
FILE=ENCFF291HHS_HCT116.bam
mv ENCFF291HHS.bam ENCFF291HHS_HCT116.bam
igvtools index $FILE

