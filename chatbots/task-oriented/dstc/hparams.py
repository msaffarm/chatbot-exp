from tensor2tensor.utils import registry

#hidden_size = 128

########################################## TRANSFORMER #################################3
# dropout = [0.2,0.7]
# layers = [2,4]
# attention heads = [1,4]
from tensor2tensor.models.transformer import transformer_tiny

@registry.register_hparams
def dstc_transformer_hparams_v1():
    hparams = transformer_tiny()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def dstc_transformer_hparams_v2():
    hparams = transformer_tiny()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams

@registry.register_hparams
def dstc_transformer_hparams_v3():
    hparams = transformer_tiny()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def dstc_transformer_hparams_v4():
    hparams = transformer_tiny()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams

@registry.register_hparams
def dstc_transformer_hparams_v5():
    hparams = transformer_tiny()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def dstc_transformer_hparams_v6():
    hparams = transformer_tiny()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams

@registry.register_hparams
def dstc_transformer_hparams_v7():
    hparams = transformer_tiny()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def dstc_transformer_hparams_v8():
    hparams = transformer_tiny()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams





########################################## LSTM #################################
from tensor2tensor.models.lstm import lstm_seq2seq, lstm_attention
# dropout = [0.9,0.7]
# layers = [2,4]
@registry.register_hparams
def dstc_lstm_hparams_v1():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 2
    hparams.batch_size = 4096
    hparams.dropout=0.9
    return hparams

@registry.register_hparams
def dstc_lstm_hparams_v2():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 2
    hparams.batch_size = 4096
    hparams.dropout=0.7
    return hparams


@registry.register_hparams
def dstc_lstm_hparams_v3():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 4
    hparams.batch_size = 4096
    hparams.dropout=0.9
    return hparams

@registry.register_hparams
def dstc_lstm_hparams_v4():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 4
    hparams.batch_size = 4096
    hparams.dropout=0.7
    return hparams


# attention LSTM
@registry.register_hparams
def dstc_lstm_attention_hparams_v1():
    hparams = lstm_attention()
    hparams.num_hidden_layers = 2
    hparams.batch_size = 4096
    hparams.dropout=0.9
    return hparams


# attention LSTM
@registry.register_hparams
def dstc_lstm_attention_hparams_v2():
    hparams = lstm_attention()
    hparams.num_hidden_layers = 2
    hparams.batch_size = 4096
    hparams.dropout=0.7
    return hparams

# attention LSTM
@registry.register_hparams
def dstc_lstm_attention_hparams_v3():
    hparams = lstm_attention()
    hparams.num_hidden_layers = 4
    hparams.batch_size = 4096
    hparams.dropout=0.9
    return hparams

# attention LSTM
@registry.register_hparams
def dstc_lstm_attention_hparams_v4():
    hparams = lstm_attention()
    hparams.num_hidden_layers = 4
    hparams.batch_size = 4096
    hparams.dropout=0.7
    return hparams





########################################## BIDIRECTTIONAL LSTM #################################
from tensor2tensor.models.lstm import lstm_seq2seq, lstm_attention
# dropout = [0.7,0.9]
# layers = [2,4]
@registry.register_hparams
def dstc_bilstm_hparams_v1():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 2
    hparams.batch_size = 4096
    hparams.dropout=0.9
    return hparams

@registry.register_hparams
def dstc_bilstm_hparams_v2():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 2
    hparams.batch_size = 4096
    hparams.dropout=0.7
    return hparams


@registry.register_hparams
def dstc_bilstm_hparams_v3():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 4
    hparams.batch_size = 4096
    hparams.dropout=0.9
    return hparams

@registry.register_hparams
def dstc_bilstm_hparams_v4():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 4
    hparams.batch_size = 4096
    hparams.dropout=0.7
    return hparams


# attention LSTM
@registry.register_hparams
def dstc_bilstm_attention_hparams_v1():
    hparams = lstm_attention()
    hparams.num_hidden_layers = 2
    hparams.batch_size = 4096
    hparams.dropout=0.9
    return hparams


# attention LSTM
@registry.register_hparams
def dstc_bilstm_attention_hparams_v2():
    hparams = lstm_attention()
    hparams.num_hidden_layers = 2
    hparams.batch_size = 4096
    hparams.dropout=0.7
    return hparams

# attention LSTM
@registry.register_hparams
def dstc_bilstm_attention_hparams_v3():
    hparams = lstm_attention()
    hparams.num_hidden_layers = 4
    hparams.batch_size = 4096
    hparams.dropout=0.9
    return hparams

# attention LSTM
@registry.register_hparams
def dstc_bilstm_attention_hparams_v4():
    hparams = lstm_attention()
    hparams.num_hidden_layers = 4
    hparams.batch_size = 4096
    hparams.dropout=0.7
    return hparams



########################################## UNIVERSAL TRANSFORMER #################################3
from tensor2tensor.models.research.universal_transformer import universal_transformer_tiny


# dropout = [0.2,0.7]
# layers = [2,4]
# attention heads = [1,4]

@registry.register_hparams
def dstc_universal_transformer_hparams_v1():
    hparams = universal_transformer_tiny()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def dstc_universal_transformer_hparams_v2():
    hparams = universal_transformer_tiny()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams

@registry.register_hparams
def dstc_universal_transformer_hparams_v3():
    hparams = universal_transformer_tiny()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def dstc_universal_transformer_hparams_v4():
    hparams = universal_transformer_tiny()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams

@registry.register_hparams
def dstc_universal_transformer_hparams_v5():
    hparams = universal_transformer_tiny()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def dstc_universal_transformer_hparams_v6():
    hparams = universal_transformer_tiny()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams

@registry.register_hparams
def dstc_universal_transformer_hparams_v7():
    hparams = universal_transformer_tiny()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def dstc_universal_transformer_hparams_v8():
    hparams = universal_transformer_tiny()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams



########################################## UNIVERSAL TRANSFORMER ACT #################################
from tensor2tensor.models.research.universal_transformer import adaptive_universal_transformer_tiny


# dropout = [0.2,0.7]
# attention heads = [1,4]
# ACT = [basic(act),accumulated,gloabal]

@registry.register_hparams
def dstc_universal_transformer_act_hparams_v1():
    hparams = adaptive_universal_transformer_tiny()
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams


@registry.register_hparams
def dstc_universal_transformer_act_hparams_v2():
    hparams = adaptive_universal_transformer_tiny()
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams

@registry.register_hparams
def dstc_universal_transformer_act_hparams_v3():
    hparams = adaptive_universal_transformer_tiny()
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def dstc_universal_transformer_act_hparams_v4():
    hparams = adaptive_universal_transformer_tiny()
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams


@registry.register_hparams
def dstc_universal_transformer_acc_hparams_v1():
    hparams = adaptive_universal_transformer_tiny()
    hparams.act_type = "accumulated"
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams


@registry.register_hparams
def dstc_universal_transformer_acc_hparams_v2():
    hparams = adaptive_universal_transformer_tiny()
    hparams.act_type = "accumulated"
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams

@registry.register_hparams
def dstc_universal_transformer_acc_hparams_v3():
    hparams = adaptive_universal_transformer_tiny()
    hparams.act_type = "accumulated"
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def dstc_universal_transformer_acc_hparams_v4():
    hparams = adaptive_universal_transformer_tiny()
    hparams.act_type = "accumulated"
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams


@registry.register_hparams
def dstc_universal_transformer_glb_hparams_v1():
    hparams = adaptive_universal_transformer_tiny()
    hparams.act_type = "global"
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams


@registry.register_hparams
def dstc_universal_transformer_glb_hparams_v2():
    hparams = adaptive_universal_transformer_tiny()
    hparams.act_type = "global"
    hparams.dropout=0.2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams

@registry.register_hparams
def dstc_universal_transformer_glb_hparams_v3():
    hparams = adaptive_universal_transformer_tiny()
    hparams.act_type = "global"
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def dstc_universal_transformer_glb_hparams_v4():
    hparams = adaptive_universal_transformer_tiny()
    hparams.act_type = "global"
    hparams.dropout=0.7
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams

######################################### NEURAL GPU #########################################3
from tensor2tensor.models.neural_gpu import neural_gpu


# dropout = [0.2,0.7]
# hidden_layers = [2,4]

@registry.register_hparams
def dstc_ngpu_v1():
    hparams = neural_gpu()
    hparams.dropout=0.2
    hparams.batch_size=4096
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_hidden_layers = 2
    return hparams


@registry.register_hparams
def dstc_ngpu_v2():
    hparams = neural_gpu()
    hparams.dropout=0.2
    hparams.batch_size=4096
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_hidden_layers = 4
    return hparams


@registry.register_hparams
def dstc_ngpu_v3():
    hparams = neural_gpu()
    hparams.dropout=0.7
    hparams.batch_size=512
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_hidden_layers = 1
    return hparams


@registry.register_hparams
def dstc_ngpu_v4():
    hparams = neural_gpu()
    hparams.dropout=0.7
    hparams.batch_size=4096
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_hidden_layers = 4
    return hparams