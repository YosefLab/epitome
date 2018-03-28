
# Runs SeqOutBias on bams. To install SeqOutBias, visit https://guertinlab.github.io/seqOutBias/seqOutBias_user_guide.pdf
for file in 'ENCFF591XCX_hepg2.bam' 'ENCFF912JKA_hela_s3.bam' 'ENCFF538GKX_K562.bam' 'ENCFF591XCX_hpeg2.bam' 'ENCFF912JKA_hela_s3.bam' 'ENCFF966NIW_gm12878.bam' 'ENCFF059BEU_h1hesc.bam' 
do
        outfile=$file\_process.sh


	echo "samtools view -h $file | awk '{if(\$3 != "chrEBV"){print \$0}}' | samtools view -Sb > $file.filtered.bam" > $outfile

	echo "samtools view -h -o $file.filtered.bam.sam $file.filtered.bam" >> $outfile

	echo "sed -i '0,/chrEBV/{/chrEBV/d;}' $file.filtered.bam.sam" >> $outfile

	echo "samtools view -S -b $file.filtered.bam.sam > $file.filtered.header.bam" >> $outfile

	echo " ~/Programs/seqOutBias_v1.1.3.bin.linux.x86_64/seqOutBias ~/genomebuilds/hg38.fa /data/yosef2/scratch/Alyssa/encode/$file.filtered.header.bam" >> $outfile

	echo "rm $file.filtered.bam.sam $file.filtered.bam" >> $outfile
done

chmod a+rwx *_process.sh
