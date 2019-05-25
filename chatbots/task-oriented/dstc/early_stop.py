import tensorflow as tf
import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def find_latest_eval(eval_logs):
    timestamp_vals = []
    for elog in eval_logs:
        timestamp = int(elog.split('.')[3])
        timestamp_vals.append(timestamp)
    min_index = timestamp_vals.index(min(timestamp_vals))

    return eval_logs[min_index]


def find_best_model(model_name):

    LOG_DIR = os.path.join(CURRENT_DIR, "t2t_train", model_name)
    EVAL_DIR = os.path.join(LOG_DIR,'eval')

    all_evals = [f for f in os.listdir(EVAL_DIR) if "events" in f]
    eval_log = find_latest_eval(all_evals)
    # EVAL_FILE = os.path.join(LOG_DIR, "")


    # read values from 
    eval_vals = []
    for e in tf.train.summary_iterator(os.path.join(EVAL_DIR, eval_log)):
        for v in e.summary.value:
            if v.tag == 'loss' or v.tag == 'accuracy':
                eval_vals.append(v.simple_value)
    # print(eval_vals)
    # read checkpoints
    checkpoints = []
    with open(os.path.join(LOG_DIR, "checkpoint"), 'r') as f:
        checkpoints = [ n.strip() for n in f.readlines()[1:] ]


    val2checkpoint = dict(zip(eval_vals,checkpoints))
    # print(val2checkpoint)


    # find the optimal checkpoint according to early stopping
    eval_early_stopping_metric_delta = 0.01
    hop_size = 2
    
    diffs = []
    optimal_step = -1
    for i in range(len(eval_vals)-hop_size):
        diff = eval_vals[i+hop_size]-eval_vals[i]
        # detect increse in eval_loss
        if diff > 0:
            optimal_step = i
            break
        if abs(diff) <= eval_early_stopping_metric_delta:
            optimal_step = i
            break

    optimal_eval_loss = eval_vals[optimal_step]
    print("Optimal eval loss is: {}".format(optimal_eval_loss))
    optimal_checkpoint = val2checkpoint[optimal_eval_loss]
    print('Optimal checkpoint is {}'.format(optimal_checkpoint))


def main():
    MODEL_BASE_NAME = "transformer-dstc_transformer_hparams"
    # MODEL_BASE_NAME = "lstm_seq2seq-dstc_lstm_hparams"
    # MODEL_BASE_NAME = "lstm_seq2seq_attention-dstc_lstm_attention_hparams"
    # MODEL_BASE_NAME = "lstm_seq2seq_attention_bidirectional_encoder-dstc_bilstm_attention_hparams"

    for v in range(1,9):
        model_name = MODEL_BASE_NAME + '_v{}'.format(v)
        print('Getting resuls for {}'.format(model_name))
        find_best_model(model_name)
        print()


if __name__ == "__main__":
    main()