# download ATAC-seq data

# k562
wget https://www.encodeproject.org/files/ENCFF335TCS/@@download/ENCFF335TCS.bam
mv ENCFF335TCS.bam ENCFF335TCS_k562_dnase.bam

wget https://www.encodeproject.org/files/ENCFF082HNP/@@download/ENCFF082HNP.bam
mv ENCFF082HNP.bam ENCFF082HNP_k562_dnase.bam

# HepG2
wget https://www.encodeproject.org/files/ENCFF097NAZ/@@download/ENCFF097NAZ.bam
mv ENCFF097NAZ.bam ENCFF097NAZ_hepG2_dnase.bam

wget https://www.encodeproject.org/files/ENCFF221TOB/@@download/ENCFF221TOB.bam
mv ENCFF221TOB.bam ENCFF221TOB_hepg2.bam

# A549
wget https://www.encodeproject.org/files/ENCFF716ZOM/@@download/ENCFF716ZOM.bam
mv ENCFF716ZOM.bam ENCFF716ZOM_A549.bam

wget https://www.encodeproject.org/files/ENCFF611WNE/@@download/ENCFF611WNE.bam
mv ENCFF611WNE.bam ENCFF611WNE_A549.bam

wget https://www.encodeproject.org/files/ENCFF658OTR/@@download/ENCFF658OTR.bam
mv ENCFF658OTR.bam ENCFF658OTR_A549.bam

# GM12878
wget https://www.encodeproject.org/files/ENCFF593WBR/@@download/ENCFF593WBR.bam
mv ENCFF593WBR.bam ENCFF593WBR_GM12878.bam

wget https://www.encodeproject.org/files/ENCFF658WKQ/@@download/ENCFF658WKQ.bam
mv ENCFF658WKQ.bamENCFF658WKQ_GM12878.bam


# IMR-90
wget https://www.encodeproject.org/files/ENCFF023JFG/@@download/ENCFF023JFG.bam
mv ENCFF023JFG.bam ENCFF023JFG_IMR90.bam

# HELA S3
wget https://www.encodeproject.org/files/ENCFF180PCI/@@download/ENCFF180PCI.bam
mv ENCFF180PCI.bam ENCFF180PCI_HELAS3.bam

# HCT 116
wget https://www.encodeproject.org/files/ENCFF391EDU/@@download/ENCFF391EDU.bam
mv ENCFF391EDU.bam ENCFF391EDU_HCT116.bam

# H1-hesc
wget https://www.encodeproject.org/files/ENCFF059BEU/@@download/ENCFF059BEU.bam
mv ENCFF059BEU.bam ENCFF059BEU_h1hesc.bam


:'
# K562
wget https://www.encodeproject.org/files/ENCFF693NGB/@@download/ENCFF693NGB.bed.gz
mv ENCFF693NGB.bed.gz ENCFF693NGB_k562_egr1.bed.gz
gunzip ENCFF693NGB_k562_egr1.bed.gz

wget https://www.encodeproject.org/files/ENCFF036GCO/@@download/ENCFF036GCO.bed.gz
mv ENCFF036GCO.bed.gz ENCFF036GCO_k562_rb1.bed.gz
gunzip ENCFF036GCO_k562_rb1.bed.gz

wget https://www.encodeproject.org/files/ENCFF294HYU/@@download/ENCFF294HYU.bed.gz
mv ENCFF294HYU.bed.gz ENCFF294HYU_k562_rbfox2.bed.gz
gunzip ENCFF294HYU_k562_rbfox2.bed.gz

wget https://www.encodeproject.org/files/ENCFF560EPF/@@download/ENCFF560EPF.bed.gz
mv ENCFF560EPF.bed.gz ENCFF560EPF_k562_znf592.bed.gz
gunzip ENCFF560EPF_k562_znf592.bed.gz
'






