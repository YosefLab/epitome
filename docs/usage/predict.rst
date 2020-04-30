Predicting binding from Chromatin Accessibility
===============================================


Once you have `trained a model <./train.html>`__, you can predict on your own cell types.
You need a peak file called from your DNase-seq or ATAC-seq data. This peak file
can be in either `bed` or `narrowpeak` format.


To get predictions on the whole genome, run:

.. code:: python

  peak_result = model.score_whole_genome(peak_file, # chromatin accessibility peak file
    output_path, # where to save results
    chrs=["chr8","chr9"]) # chromosomes you would like to score. Leave blank for whole genome.

**Note:** Scoring on the whole genome scores about 7 million regions and takes about 1.5 hours.

TODO: talk about including histone modification files.


You can also get predictions on specific genomic regions:

.. code:: python

  results = model.score_peak_file(peak_file, # chromatin accessibility peak file
    regions_file) # bed file of regions to score

This method returns a dataframe of the scored predictions.
