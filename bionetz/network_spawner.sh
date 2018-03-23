for offset in 0 100
do
	for weight in offset+1 offset+25 offset+50 offset+75
	do
		python3 main.py --tfrecords\ 
		--train /data/epitome/output/training/\ 
		--valid /data/epitome/output/valid/\ 
		--name weighted$weight\ 
		--rate 1e-5\ 
		--valid_size 34375\ 
		--pos_weight $weight\ 
		--batch 32\ 
		--logdir /data/jwhughes/logs\ 
		--iterations 100000
	done
	wait
done