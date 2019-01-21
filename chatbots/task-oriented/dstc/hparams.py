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
from tensor2tensor.models.lstm import lstm_seq2seq, lstm_attention_base
# dropout = [0.2,0.7]
# layers = [2,4]
@registry.register_hparams
def dstc_lstm_hparams_v1():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.2
    return hparams

@registry.register_hparams
def dstc_lstm_hparams_v2():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.7
    return hparams


@registry.register_hparams
def dstc_lstm_hparams_v3():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.2
    return hparams

@registry.register_hparams
def dstc_lstm_hparams_v4():
    hparams = lstm_seq2seq()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.7
    return hparams


# attention LSTM
@registry.register_hparams
def dstc_lstm_attention_hparams_v1():
    hparams = lstm_attention_base()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.2
    return hparams


# attention LSTM
@registry.register_hparams
def dstc_lstm_attention_hparams_v2():
    hparams = lstm_attention_base()
    hparams.num_hidden_layers = 2
    hparams.dropout=0.7
    return hparams

# attention LSTM
@registry.register_hparams
def dstc_lstm_attention_hparams_v3():
    hparams = lstm_attention_base()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.2
    return hparams

# attention LSTM
@registry.register_hparams
def dstc_lstm_attention_hparams_v4():
    hparams = lstm_attention_base()
    hparams.num_hidden_layers = 4
    hparams.dropout=0.7
    return hparams



# ########################################## UNIVERSAL TRANSFORMER #################################3
# from tensor2tensor.models.transformer import transformer_tiny


# # dropout = [0.2,0.7]
# # layers = [2,4]
# # attention heads = [1,4]

# @registry.register_hparams
# def dstc_transformer_hparams_v1():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 2
#     hparams.dropout=0.2
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 1
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v2():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 2
#     hparams.dropout=0.2
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 4
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v3():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 4
#     hparams.dropout=0.2
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 1
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v4():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 4
#     hparams.dropout=0.2
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 4
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v5():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 2
#     hparams.dropout=0.7
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 1
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v6():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 2
#     hparams.dropout=0.7
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 4
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v7():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 4
#     hparams.dropout=0.7
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 1
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v8():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 4
#     hparams.dropout=0.7
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 4
#     return hparams





# ########################################## UNIVERSAL TRANSFORMER ACT #################################3
# # dropout = [0.2,0.7]
# # layers = [2,4]
# # attention heads = [1,4]

# @registry.register_hparams
# def dstc_transformer_hparams_v1():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 2
#     hparams.dropout=0.2
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 1
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v2():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 2
#     hparams.dropout=0.2
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 4
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v3():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 4
#     hparams.dropout=0.2
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 1
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v4():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 4
#     hparams.dropout=0.2
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 4
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v5():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 2
#     hparams.dropout=0.7
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 1
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v6():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 2
#     hparams.dropout=0.7
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 4
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v7():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 4
#     hparams.dropout=0.7
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 1
#     return hparams

# @registry.register_hparams
# def dstc_transformer_hparams_v8():
#     hparams = transformer_tiny()
#     hparams.num_hidden_layers = 4
#     hparams.dropout=0.7
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.num_heads = 4
#     return hparams
