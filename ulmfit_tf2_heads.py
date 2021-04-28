import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
from .awdlstm_tf2 import *
from .ulmfit_tf2 import *

def ulmfit_sequence_tagger(*, num_classes=3, pretrained_weights=None, fixed_seq_len=None, spm_model_file,
                              also_return_spm_encoder=False):
    print("Building model from Python code (not tf.saved_model)...")
    _, enc_num, _, spm_encoder_model = tf2_ulmfit_encoder(fixed_seq_len=fixed_seq_len, spm_model_file=spm_model_file)
    if pretrained_weights is not None:
        print("Restoring weights from file....")
        enc_num.load_weights(pretrained_weights)
    else:
        print("!!! THE MODEL WEIGHTS ARE UNINITIALIZED !!! Make sure to restore them from file.")
    print(f"Adding sequence tagging head with n_classes={num_classes}")
    tagger_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(enc_num.output)
    tagger = tf.keras.Model(inputs=enc_num.inputs, outputs=tagger_head)
    if also_return_spm_encoder is True:
        return tagger, spm_encoder_model
    else:
        return tagger

def ulmfit_fake_tagger(*, num_classes=3, pretrained_weights=None, fixed_seq_len=None, spm_model_file,
                          also_return_spm_encoder=False):
    print("Building a regular LSTM model using only standard Keras blocks...")
    AWD_LSTM_Cell1 = WeightDropLSTMCell(1152, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    # AWD_LSTM_Cell1 = tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform', dropout=0.5)
    fake_model = tf.keras.models.Sequential([
        tf.keras.layers.Input((fixed_seq_len,)),
        tf.keras.layers.Masking(mask_value=1),
        tf.keras.layers.Embedding(35000, 400),
        EmbeddingDropout(encoder_dp_rate=0.4, name="emb_dropout"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.SpatialDropout1D(0.3),
        #tf.keras.layers.LSTM(1152, return_sequences=True),
        tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1"),
        tf.keras.layers.SpatialDropout1D(0.5),
        tf.keras.layers.LSTM(1152, return_sequences=True),
        tf.keras.layers.SpatialDropout1D(0.5),
        tf.keras.layers.LSTM(400, return_sequences=True),
        tf.keras.layers.SpatialDropout1D(0.5),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation='softmax'))
        ])
    return fake_model
