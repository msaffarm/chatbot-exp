PROBLEM=dstc_problem


# MODEL=transformer
# HPARAMS=transformer_base_single_gpu
MODEL=universal_transformer
# HPARAMS=universal_transformer_tiny
# HPARAMS=universal_transformer_skip_tiny
# MODEL=transformer
# HPARAMS=transformer_tiny
# MODEL=universal_transformer
HPARAMS=adaptive_universal_transformer_tiny
# MODEL=lstm_seq2seq_attention
# HPARAMS=lstm_attention

DATASET="Rest"
CURRENT_DIR=$PWD
DATA_DIR=$CURRENT_DIR/t2t_data
TMP_DIR=$CURRENT_DIR/t2t_datagen
# TRAIN_DIR=$CURRENT_DIR/t2t_train/$PROBLEM/$MODEL-$HPARAMS
TRAIN_DIR=$CURRENT_DIR/t2t_train/$PROBLEM/$DATASET/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# DECODE_FILE=$CURRENT_DIR/sample-input-test.txt
# DECODE_FILE=$CURRENT_DIR/$DATASET-input-test.txt
DECODE_FILE=$CURRENT_DIR/DSTC2input-test.txt


BEAM_SIZE=4
ALPHA=0.6
# --checkpoint_path=$TRAIN_DIR/model.ckpt-2000 \

t2t-decoder \
  --t2t_usr_dir=$CURRENT_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=$DATASET-response.txt \
  --checkpoint_path=$TRAIN_DIR/model.ckpt-5000
