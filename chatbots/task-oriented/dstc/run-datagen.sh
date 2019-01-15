PROBLEM=dstc_problem

CURRENT_DIR=$PWD
DATA_DIR=$CURRENT_DIR/t2t_data
TMP_DIR=$CURRENT_DIR/t2t_datagen

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
# t2t-trainer --t2t_usr_dir=$CURRENT_DIR --registry_help

t2t-datagen \
  --t2t_usr_dir=$CURRENT_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM