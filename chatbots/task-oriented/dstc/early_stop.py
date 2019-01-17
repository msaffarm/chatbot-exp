import tensorflow as tf
import os


MODEL_NAME = "transformer-m2m_m_transformer_hparams"

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, "t2t_train", MODEL_NAME)
EVAL_DIR = os.path.join(LOG_DIR,'eval')

eval_log = [f for f in os.listdir(EVAL_DIR) if "events" in f][0]
# EVAL_FILE = os.path.join(LOG_DIR, "")



# read values from 
eval_vals = []
for e in tf.train.summary_iterator(os.path.join(EVAL_DIR, eval_log)):
    for v in e.summary.value:
        if v.tag == 'loss' or v.tag == 'accuracy':
            eval_vals.append(v.simple_value)

# read checkpoints
checkpoints = []
with open(os.path.join(LOG_DIR, "checkpoint"), 'r') as f:
    checkpoints = [ n.strip() for n in f.readlines()[1:] ]

# print(len(eval_vals))
# print(len(checkpoints))

val2checkpoint = dict(zip(eval_vals,checkpoints))
# print(val2checkpoint)


# find the optimal checkpoint according to early stopping
eval_early_stopping_metric_delta = 0.099
diffs = []
optimal_step = -1
for i in range(len(eval_vals)-1):
    diff = eval_vals[i+1]-eval_vals[i]
    # detect increse in eval_loss
    if diff > 0:
        optimal_step = i
        break
    if abs(diff) <= eval_early_stopping_metric_delta:
        optimal_step = i
        break

optimal_eval_loss = eval_vals[optimal_step]
print(optimal_eval_loss)
optimal_checkpoint = val2checkpoint[optimal_eval_loss]
print(optimal_checkpoint)
