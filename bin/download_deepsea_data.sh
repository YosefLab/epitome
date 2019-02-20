#!/bin/bash


# requires output location to download data to 
LOCATION=$1

wget -P ${LOCATION} http://deepsea.princeton.edu/media/code/deepsea_train.v0.9.tar.gz 
tar -C ${LOCATION} -zxvf "${LOCATION}/deepsea_train.v0.9.tar.gz"
