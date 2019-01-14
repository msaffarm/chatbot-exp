# PROBLEM=m2_m_problem
PROBLEM=dstc_problem

# MODEL=transformer
# HPARAMS=transformer_tiny

# MODEL=universal_transformer
# HPARAMS=universal_transformer_highway_tiny
# HPARAMS=universal_transformer_tiny

MODEL=universal_transformer
HPARAMS=adaptive_universal_transformer_tiny

# MODEL=lstm_seq2seq_attention
# HPARAMS=lstm_attention

# MODEL=lstm_seq2seq_attention_bidirectional_encoder
# HPARAMS=lstm_attention

DATASET="Rest"
CURRENT_DIR=$PWD
DATA_DIR=$CURRENT_DIR/t2t_data
TMP_DIR=$CURRENT_DIR/t2t_datagen
TRAIN_DIR=$CURRENT_DIR/t2t_train/$PROBLEM/$DATASET/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Train
t2t-trainer \
  --t2t_usr_dir=$CURRENT_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --train_steps=20000 \
  --eval_steps=999999 \
  --eval_early_stopping_steps=1000 \
  --local_eval_frequency=500 \
  --eval_throttle_seconds=1 \
  --keep_checkpoint_max=20
