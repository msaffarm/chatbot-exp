PROBLEM=dstc_problem

MODEL=transformer
declare -a hparams=("dstc_transformer_hparams_v1"
                    "dstc_transformer_hparams_v2"
                    "dstc_transformer_hparams_v3"
                    "dstc_transformer_hparams_v4"
                    "dstc_transformer_hparams_v5"
                    "dstc_transformer_hparams_v6"
                    "dstc_transformer_hparams_v7"
                    "dstc_transformer_hparams_v8"
                    )

# HPARAMS=m2m_m_transformer_hparams
# HPARAMS_RANGE=transformer_base_range

# MODEL=universal_transformer
# HPARAMS=universal_transformer_highway_tiny
# HPARAMS=universal_transformer_tiny

# MODEL=universal_transformer
# HPARAMS=adaptive_universal_transformer_tiny

# MODEL=lstm_seq2seq_attention
# HPARAMS=lstm_attention

# MODEL=lstm_seq2seq_attention_bidirectional_encoder
# HPARAMS=lstm_attention

CURRENT_DIR=$PWD
DATA_DIR=$CURRENT_DIR/t2t_data
TMP_DIR=$CURRENT_DIR/t2t_datagen
TRAIN_DIR=$CURRENT_DIR/t2t_train/$MODEL-$HPARAMS

# mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# # Train
# t2t-trainer \
#   --t2t_usr_dir=$CURRENT_DIR \
#   --data_dir=$DATA_DIR \
#   --problem=$PROBLEM \
#   --model=$MODEL \
#   --hparams_set=$HPARAMS \
#   --output_dir=$TRAIN_DIR \
#   --train_steps=15000 \
#   --eval_steps=999999 \
#   --local_eval_frequency=200 \
#   --eval_throttle_seconds=1 \
#   --keep_checkpoint_max=30


# Train loop
for hparam in "${hparams[@]}"
do
  echo "$hparam"
  HPARAMS=$hparam
  TRAIN_DIR=$CURRENT_DIR/t2t_train/$MODEL-$HPARAMS
  mkdir -p $TRAIN_DIR
  
  t2t-trainer \
    --t2t_usr_dir=$CURRENT_DIR \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --train_steps=6000 \
    --eval_steps=999999 \
    --local_eval_frequency=300 \
    --eval_throttle_seconds=1 \
    --keep_checkpoint_max=20

done