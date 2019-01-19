PROBLEM=dstc_problem


# MODEL=universal_transformer
# HPARAMS=adaptive_universal_transformer_tiny


# MODEL=transformer
# HPARAMS=m2m_m_transformer_hparams


# HPARAMS=transformer_base_single_gpu
# HPARAMS=universal_transformer_tiny
# HPARAMS=universal_transformer_skip_tiny
# MODEL=transformer
# MODEL=universal_transformer

# MODEL=lstm_seq2seq_attention
# HPARAMS=lstm_attention

MODEL=lstm_seq2seq_attention
HPARAMS=lstm_attention



CURRENT_DIR=$PWD
DATA_DIR=$CURRENT_DIR/t2t_data
TMP_DIR=$CURRENT_DIR/t2t_datagen
TRAIN_DIR=$CURRENT_DIR/t2t_train/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# DECODE_FILE=$CURRENT_DIR/sample-input-test.txt
# DECODE_FILE=$CURRENT_DIR/$DATASET-input-test.txt
DECODE_FILE=$CURRENT_DIR/test/DSTC2input-test.txt
DECODE_PATH=$CURRENT_DIR/test

# BEAM_SIZE=4
# ALPHA=0.6
# --checkpoint_path=$TRAIN_DIR/model.ckpt-2000 \
# --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \

t2t-decoder \
  --t2t_usr_dir=$CURRENT_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=$DECODE_PATH/$MODEL-$HPARAMS-response.txt \
  --checkpoint_path=$TRAIN_DIR/model.ckpt-12000
