#!/bin/bash

FWDIR="$(cd `dirname $0`/..; pwd)"

DATE=`date +"%Y_%m_%d_%H_%M_%S"`
PIPELINE=net.akmorrow13.epitome.Epitome
$FWDIR/bin/run-pipeline-yarn.sh $PIPELINE $FWDIR/target/scala-2.10/epitome-assembly-0.1.jar "$@" 2>&1