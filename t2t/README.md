# Tensor2tensor

### Generating TF Records
* Replace `USR_DIR`, `DATA_DIR`, `TMP_DIR`, `PROBLEM` in `run.bash`. 
  * `USR_DIR` should be the path to this folder, which contains the `__init__.py` file.
  * `DATA_DIR` is where the TF records will be created.
  * `TMP_DIR` is where DeepSEA data will be downloaded. If you already have DeepSEA data downloaded, set `TMP_DIR` to that location to avoid downloading another copy.
  * `PROBLEM` is set to deep_sea_problem by default. Set `PROBLEM` to `epitome_problem` if desired. 
* When variables have all been updated, run `./run.bash`.

### Training
* TODO
