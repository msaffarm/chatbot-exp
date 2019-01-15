PROBLEM=m2_m_problem
# MODEL=transformer
# HPARAMS=transformer_base_single_gpu
MODEL=universal_transformer
HPARAMS=adaptive_universal_transformer_tiny

DATASET="Movie"
CURRENT_DIR=$PWD
DATA_DIR=$CURRENT_DIR/t2t_data
TMP_DIR=$CURRENT_DIR/t2t_datagen
# TRAIN_DIR=$CURRENT_DIR/t2t_train/$PROBLEM/$MODEL-$HPARAMS
TRAIN_DIR=$CURRENT_DIR/t2t_train/$PROBLEM/$DATASET/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --t2t_usr_dir=$CURRENT_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_interactive