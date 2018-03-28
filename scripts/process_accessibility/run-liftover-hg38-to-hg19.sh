# lifts hg38 genome to h16. Requires liftover program and hg38ToHg19 chain file
for file in 'ENCFF059BEU_h1hesc.bam.filtered.header_scaled.bed' 'ENCFF591XCX_hepg2.bam.filtered.header_scaled.bed' 'ENCFF473CCA_A549_DNase_filtered_header_scaled.bed'  'ENCFF912JKA_hela_s3.bam_filtered_header_scaled.bed' 'ENCFF538GKX_K562.bam.filtered.header_scaled.bed' 'ENCFF966NIW_gm12878.bam.filtered.header_scaled.bed'
do
	~/Programs/liftOver $file ~/Programs/liftOverChains/hg38ToHg19.over.chain.gz $file.hg19.bed $file.hg19unlifted.bed
done
