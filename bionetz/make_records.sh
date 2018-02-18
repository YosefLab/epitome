#!/bin/bash


let total=1000
let numshards=$1
for cell in 'HeLa-S3' 'GM12878' 'H1-hESC' 'HepG2' 'K562'
do
	for i in `seq 1 $numshards`;
	do
		let step=$total/$numshards
		let start=(i-1)*$step
		let stop=i*$step
		python3 make_tf_records.py --start $start --stop $stop --cell $cell &
		echo $cell $i
	done
	wait
done
