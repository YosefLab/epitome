# Epitome

Full pipeline for learning TFBS from epigenetic datasets.

## Modules:
- python
-- preprocess: contains code to preprocess raw datasets (bam files) to featurized datasets
-- models: contains learning models trained from preprocessed datasets
- assembly
-- contains assembly code to package jar for running the notebook


Requirements:
- maven
- python
- Spark 1.6.3

Clean:
mvn clean


Package:
mvn package -DskipTests


Test:
mvn test

Run notebook:
./bin/run-notebook

There is a demo notebook file in python/notebooks that demonstrates
loading in feature and alignment files using ADAM.