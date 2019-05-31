PROBLEM=dstc_problem


# MODEL=universal_transformer
# HPARAMS=adaptive_universal_transformer_tiny


# MODEL=transformer
# HPARAMS=dstc_transformer_hparams_v4
# DESIRED_CHECKPOINT=3600

# MODEL=lstm_seq2seq
# HPARAMS=dstc_lstm_hparams_v2
# DESIRED_CHECKPOINT=6300

# MODEL=lstm_seq2seq_attention
# HPARAMS=dstc_lstm_attention_hparams_v2
# DESIRED_CHECKPOINT=6300

# MODEL=lstm_seq2seq_attention_bidirectional_encoder
# HPARAMS=dstc_bilstm_attention_hparams_v2
# DESIRED_CHECKPOINT=6300

# MODEL=lstm_seq2seq_bidirectional_encoder
# HPARAMS=dstc_bilstm_hparams_v2
# DESIRED_CHECKPOINT=6300

# MODEL=universal_transformer
# HPARAMS=dstc_universal_transformer_hparams_v2
# DESIRED_CHECKPOINT=4200

MODEL=universal_transformer
HPARAMS=dstc_universal_transformer_glb_hparams_v4
DESIRED_CHECKPOINT=3900





CURRENT_DIR=$PWD
DATA_DIR=$CURRENT_DIR/t2t_data
TMP_DIR=$CURRENT_DIR/t2t_datagen
TRAIN_DIR=$CURRENT_DIR/t2t_train/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# DECODE_FILE=$CURRENT_DIR/test/DSTC2input-test.txt
# DECODE_PATH=$CURRENT_DIR/test

DECODE_FILE=$CURRENT_DIR/dev/DSTC2input-dev.txt
DECODE_PATH=$CURRENT_DIR/dev

# --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \

BEAM_SIZE=2
ALPHA=0.0

t2t-decoder \
  --t2t_usr_dir=$CURRENT_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --output_dir=$TRAIN_DIR \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=$DECODE_PATH/$MODEL-$HPARAMS-$DESIRED_CHECKPOINT-$BEAM_SIZE-$ALPHA-response.txt \
  --checkpoint_path=$TRAIN_DIR/model.ckpt-$DESIRED_CHECKPOINT
