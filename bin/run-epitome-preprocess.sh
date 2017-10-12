#!/bin/bash

FWDIR="$(cd `dirname $0`/..; pwd)"

DATE=`date +"%Y_%m_%d_%H_%M_%S"`
PIPELINE=net.akmorrow13.epitome.Epitome

$FWDIR/bin/epitome-submit.sh $PIPELINE \
--num-executors 10 --executor-memory 40g \
--packages org.apache.parquet:parquet-avro:1.8.1 -- \
/data/anv/DREAMDATA/epitome_data/ENCFF343HTD_k562_dnase.bam \
/data/anv/DREAMDATA/epitome_data/ENCFF036GCO_k562_rb1.bed,/data/anv/DREAMDATA/epitome_data/ENCFF294HYU_k562_rbfox2.bed,/data/anv/DREAMDATA/epitome_data/ENCFF560EPF_k562_znf592.bed,/data/anv/DREAMDATA/epitome_data/ENCFF693NGB_k562_egr1.bed \
RB1,RBFOX2,ZNF592,EGR1 \
/home/eecs/akmorrow/ADAM/epitome_out
