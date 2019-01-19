from tensor2tensor.utils import metrics, registry
from tensor2tensor.models.transformer import transformer_tiny


# dropout = [0.2,0.7]
# layers = [2,4]
# attention heads = [1,4]
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




# def transformer_base_v1():
#   """Set of hyperparameters."""
#   hparams = common_hparams.basic_params1()
#   hparams.norm_type = "layer"
#   hparams.hidden_size = 512
#   hparams.batch_size = 4096
#   hparams.max_length = 256
#   hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
#   hparams.optimizer_adam_epsilon = 1e-9
#   hparams.learning_rate_schedule = "legacy"
#   hparams.learning_rate_decay_scheme = "noam"
#   hparams.learning_rate = 0.1
#   hparams.learning_rate_warmup_steps = 4000
#   hparams.initializer_gain = 1.0
#   hparams.num_hidden_layers = 6
#   hparams.initializer = "uniform_unit_scaling"
#   hparams.weight_decay = 0.0
#   hparams.optimizer_adam_beta1 = 0.9
#   hparams.optimizer_adam_beta2 = 0.98
#   hparams.num_sampled_classes = 0
#   hparams.label_smoothing = 0.1
#   hparams.shared_embedding_and_softmax_weights = True
#   hparams.symbol_modality_num_shards = 16

#   # Add new ones like this.
#   hparams.add_hparam("filter_size", 2048)
#   # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
#   hparams.add_hparam("num_encoder_layers", 0)
#   hparams.add_hparam("num_decoder_layers", 0)
#   # Attention-related flags.
#   hparams.add_hparam("num_heads", 8)
#   hparams.add_hparam("attention_key_channels", 0)
#   hparams.add_hparam("attention_value_channels", 0)
#   hparams.add_hparam("ffn_layer", "dense_relu_dense")
#   hparams.add_hparam("parameter_attention_key_channels", 0)
#   hparams.add_hparam("parameter_attention_value_channels", 0)
#   # All hyperparameters ending in "dropout" are automatically set to 0.0
#   # when not in training mode.
#   hparams.add_hparam("attention_dropout", 0.0)
#   hparams.add_hparam("attention_dropout_broadcast_dims", "")
#   hparams.add_hparam("relu_dropout", 0.0)
#   hparams.add_hparam("relu_dropout_broadcast_dims", "")
#   hparams.add_hparam("pos", "timing")  # timing, none
#   hparams.add_hparam("nbr_decoder_problems", 1)
#   hparams.add_hparam("proximity_bias", False)
#   hparams.add_hparam("use_pad_remover", True)
#   hparams.add_hparam("self_attention_type", "dot_product")
#   hparams.add_hparam("max_relative_position", 0)
#   hparams.add_hparam("conv_first_kernel", 3)
#   # These parameters are only used when ffn_layer=="local_moe_tpu"
#   hparams.add_hparam("moe_overhead_train", 1.0)
#   hparams.add_hparam("moe_overhead_eval", 2.0)
#   hparams.moe_num_experts = 16
#   hparams.moe_loss_coef = 1e-3
#   return hparams
