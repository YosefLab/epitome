# download ATAC-seq data
wget https://www.encodeproject.org/files/ENCFF343HTD/@@download/ENCFF343HTD.bam
mv ENCFF343HTD.bam ENCFF343HTD_k562_dnase.bam

# download data for 4 TFs
wget https://www.encodeproject.org/files/ENCFF693NGB/@@download/ENCFF693NGB.bed.gz
mv ENCFF693NGB.bed.gz ENCFF693NGB_k562_egr1.bed.gz
gunzip ENCFF693NGB_k562_egr1.bed.gz

wget https://www.encodeproject.org/files/ENCFF036GCO/@@download/ENCFF036GCO.bed.gz
mv ENCFF036GCO.bed.gz ENCFF036GCO_k562_rb1.bed.gz
gunzip ENCFF036GCO_k562_rb1.bed.gz

wget https://www.encodeproject.org/files/ENCFF294HYU/@@download/ENCFF294HYU.bed.gz
mv ENCFF294HYU.bed.gz ENCFF294HYU_k562_rbfox2.bed.gz
gunzip ENCFF036GCO_k562_rbfox2.bed.gz

wget https://www.encodeproject.org/files/ENCFF560EPF/@@download/ENCFF560EPF.bed.gz
mv ENCFF560EPF.bed.gz ENCFF560EPF_k562_znf592.bed.gz
gunzip ENCFF560EPF_k562_znf592.bed.gz
