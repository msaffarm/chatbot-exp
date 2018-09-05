PROBLEM=m2_m_problem
# MODEL=transformer
MODEL=lstm_seq2seq_attention
# HPARAMS=transformer_base_single_gpu
HPARAMS=my_lstm_params_set1

CURRENT_DIR=$PWD
DATA_DIR=$CURRENT_DIR/t2t_data
TMP_DIR=$CURRENT_DIR/t2t_datagen
TRAIN_DIR=$CURRENT_DIR/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --t2t_usr_dir=$CURRENT_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --train_steps=30000 \
  --eval_early_stopping_steps=500 \
  --local_eval_frequency=1000 \
  --eval_steps=999999 \
  --eval_throttle_seconds=1 \
  --keep_checkpoint_max=100 \
  --eval_early_stopping_metric_delta=0.1 \
  --schedule=train