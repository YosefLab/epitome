#!/bin/bash
set +ex

# Downloads ENCODE DNase data for 12 cell types
# # Requirements
# - IGVTools: (used for indexing bam files)
# -- https://software.broadinstitute.org/software/igv/download
# -- then add as alias to bash (alias igvtools=<path_to_igvtools>

# Requires argument of PATH to IGVTOOLS
IGVTOOLS_PATH="$1"

# Define ENCODE ID and celltype name, separated by "_"
PREFIXES=( "ENCFF414MBW_A549" "ENCFF678VYF_K562" "ENCFF571SSA_H1heSC" "ENCFF775ZJX_GM12878" "ENCFF224FMI_HepG2"  "ENCFF783TMX_HeLaS3" "ENCFF757PTA_HUVEC" "ENCFF070BAN_GM12891" "ENCFF322PZC_MCF7" "ENCFF260LKE_GM12892" "ENCFF291HHS_HCT116" )

for F in "${PREFIXES[@]}"
do
    PREFIX=`echo $F | cut -d \_ -f 1`
    CELLTYPE=`echo $F | cut -d \_ -f 2`

    FILE="${PREFIX}_${CELLTYPE}.bam"
    SORTEDFILE="${PREFIX}_sorted_${CELLTYPE}.bam"
    
    # Download file
    DOWNLOAD_URL=https://www.encodeproject.org/files/$PREFIX/@@download/$PREFIX.bam
    wget $DOWNLOAD_URL
    
    # Add celltype prefix so we know which celltype it is
    mv $PREFIX.bam $FILE
    
    # Sort bam
    $IGVTOOLS_PATH sort $FILE $SORTEDFILE
    
    # rm unsorted file
    rm $FILE
    
    # Index sorted bam
    $IGVTOOLS_PATH index $SORTEDFILE

    echo "" # newline
done

echo "done processing DNase files"
