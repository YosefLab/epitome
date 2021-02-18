Predicting peaks for ChIP-seq targets
=====================================


Once you have `trained a model <./train.html>`__, you can predict on your own cell types.
You need a peak file called from your DNase-seq or ATAC-seq data. This peak file
can be in either `bed` or `narrowpeak` format.


To get predictions on the whole genome, run:

.. code:: python

  peak_result = model.score_whole_genome(peak_files, # list of bed files containing similarity data (either chromatin accessibility, histone modifications, or other)
    output_path, # where to save results
    chrs=["chr8","chr9"]) # chromosomes you would like to score. Leave blank to score the whole genome whole genome.

**Note:** Scoring on the whole genome scores about 7 million regions and takes about 1.5 hours.

Using histone modifications to compute cell type similarity
-----------------------------------------------------------
``peak_files`` is a list of bed or narrowpeak files. Each file represents a different
assay from your cell type of interest that is used to compute cell type similarity.
If you just use DNase-seq to compute cell type similarity, ``peak_files`` should be a single
bed file of either ATAC-seq or DNase-seq peaks. If you use additional assays to compute
cell type similarity, such as histone modifications, you should include a separate bed file
for each assay used in computing cell type similarity.

For example, the following example trains an Epitome model using DNase-seq and H3K27ac to compute cell type
similarity, and then predicts using the ``score_whole_genome`` function:

.. code:: python

  # define the dataset, using DNase-seq and H3K27ac to compute similarity
  targets = ['CTCF', 'RAD21', 'SMC3']
  dataset = EpitomeDataset(targets, similarity_targets=['DNase', 'H3K27ac'])

  # create and train model
  model = EpitomeModel(dataset)
  model.train(5000)

  # list of paths to bed files for similarity assays for a cell type of interest
  peak_files = ['my_DNase_peaks.bed', 'my_H3K27ac_peaks.bed']

  peak_result = model.score_whole_genome(peak_files, # list of bed files containing similarity data (either chromatin accessibility, histone modifications, or other)
    output_path, # where to save results
    chrs=["chr8","chr9"]) # chromosomes you would like to score. Leave blank to score the whole genome whole genome.


You can also get predictions on specific genomic regions:

.. code:: python

  results = model.score_peak_file(peak_files, # chromatin accessibility peak file
    regions_file) # bed file of regions to score

This method returns a dataframe of the scored predictions.
