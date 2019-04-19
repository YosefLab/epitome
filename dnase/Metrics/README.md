The metrics fold contains code for running and comparing to other methods.

Comparative methods include:
1. DeepSEA (run through Kipoi)
2. DAStk   


Overlaps analysis:
Overlaps analysis runs a motif database scan and scores predictions.
To run overlaps analysis, first run: 
1. save_overlaps_results.py: Counts motifs from a motif directory in 
a set of test regions.
2. Run-Overlaps.ipynb: Gathers metrics (ROC/PR) for accuracy

This analysis is currently not included because it is doing so poorly.