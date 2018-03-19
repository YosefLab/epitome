USR_DIR=/Users/Gunjan/Documents/BDG/epitome/epitome/t2t
PROBLEM=deep_sea_problem
DATA_DIR=/Users/Gunjan/Documents/BDG/epitome/epitome/t2t/data
TMP_DIR=/Users/Gunjan/Documents/BDG/epitome
mkdir -p $DATA_DIR $TMP_DIR

t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM