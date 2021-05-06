import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.keras.engine import keras_tensor
from .awdlstm_tf2 import *
from .ulmfit_tf2 import *

def ulmfit_rnn_encoder_native(*, pretrained_weights=None, fixed_seq_len=None, spm_model_args,
                              also_return_spm_encoder=False, use_awd=True):
    print("Building model from Python code (not tf.saved_model)...")
    lm_num, enc_num, _, spm_encoder_model = tf2_ulmfit_encoder(fixed_seq_len=fixed_seq_len, spm_args=spm_model_args,
                                                               use_awd=use_awd,
                                                               flatten_ragged_outputs=False)
    if pretrained_weights is not None:
        print("Restoring weights from file....")
        lm_num.load_weights(pretrained_weights)
    else:
        print("!!! THE MODEL WEIGHTS ARE UNINITIALIZED !!! Make sure to restore them from file.")
    if also_return_spm_encoder is True:
        return enc_num, spm_encoder_model
    else:
        return enc_num

def ulmfit_rnn_encoder_hub(*, pretrained_weights=None, fixed_seq_len=None, spm_model_args,
                              also_return_spm_encoder=False, use_awd=True):
    print(f"Info: When restoring the ULMFiT encoder from a SavedModel, `use_awd` and `spm_model_args` have no effect`")
    restored_hub = hub.load(pretrained_weights)
    il = tf.keras.layers.Input(shape=(None,), ragged=True if fixed_seq_len is None else False, name="numericalized_input", dtype=tf.int32)
    kl = hub.KerasLayer(restored_hub.encoder_num, trainable=True, name="ulmfit_encoder")(il)
    if fixed_seq_len == None: # TODO: check tensorspec and print a pretty info if ragged parameters here and in the SavedModel are incompatible
        rec_ragged_tensor = tf.RaggedTensor.from_row_splits(kl[0], kl[1])
        model = tf.keras.models.Model(inputs=il, outputs=rec_ragged_tensor)
    else:
        model = tf.keras.models.Model(inputs=il, outputs=kl)
        #model = tf.keras.models.Model(inputs=il, outputs=kl)
    if also_return_spm_encoder:
        return model, restored_hub.spm_encoder_model
    else:
        return model

def ulmfit_sequence_tagger(*, enc_num, num_classes=3):
    print(f"Adding sequence tagging head with n_classes={num_classes}")
    tagger_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(enc_num.output)
    tagger_model = tf.keras.models.Model(inputs=enc_num.inputs, outputs=tagger_head)
    return tagger_model

############# THE CODE BELOW THIS LINE IS FOR DEBUGGING / UNSTABLE / EXPERIMENTAL ###########

# def ulmfit_baseline_tagger(*, num_classes=3, pretrained_weights=None, fixed_seq_len=None, spm_model_file,
#                           also_return_spm_encoder=False):
#     print("Building a regular LSTM model using only standard Keras blocks...")
#     # AWD_LSTM_Cell1 = WeightDropLSTMCell(1152, kernel_initializer='glorot_uniform', weight_dropout=0.5)
#     if fixed_seq_len is None:
#         raise NotImplementedError("Variable sequence length training not implemented for baseline model yet")
#     else:
#         fake_model = tf.keras.models.Sequential([
#             tf.keras.layers.Input((fixed_seq_len,)),
#             #tf.keras.layers.Masking(mask_value=1),
#             CustomMaskableEmbedding(35000, 400, mask_value=1, name="ulmfit_embeds"),
#             # tf.keras.layers.Embedding(35000, 400),
#             EmbeddingDropout(encoder_dp_rate=0.4, name="emb_dropout"),
#             tf.keras.layers.Dropout(0.3),
#             tf.keras.layers.SpatialDropout1D(0.3),
#             tf.keras.layers.LSTM(1152, return_sequences=True, name="Plain_LSTM1"),
#             #tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1"),
#             tf.keras.layers.SpatialDropout1D(0.5),
#             tf.keras.layers.LSTM(1152, return_sequences=True, name="Plain_LSTM2"),
#             tf.keras.layers.SpatialDropout1D(0.5),
#             tf.keras.layers.LSTM(400, return_sequences=True, name="Plain_LSTM3"),
#             tf.keras.layers.SpatialDropout1D(0.5),
#             tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation='softmax'), name="tagger_head")
#             ])
#     if pretrained_weights is not None:
#         print("Restoring weights from file... (observe the warnings!)")
#         fake_model.load_weights(pretrained_weights)
#     else:
#         print("!!! THE MODEL WEIGHTS ARE UNINITIALIZED !!! You are training the model without embeddings - it *will* overfit.")
#     return fake_model

def ulmfit_encoder_sequential(*, pretrained_weights=None, fixed_seq_len=None):
    print("Building an AWD-LSTM model using Keras Sequential API")
    # outside_scope_variable = tf.Variable(tf.ones((1152,4608), dtype=tf.float32), trainable=False)
    AWD_LSTM_Cell1 = WeightDropLSTMCell(1152, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    AWD_LSTM_Cell2 = WeightDropLSTMCell(1152, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    AWD_LSTM_Cell3 = WeightDropLSTMCell(400, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    # AWD_LSTM_Cell1 = tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform', dropout=0.5)
    base_model = tf.keras.models.Sequential([
        tf.keras.layers.Input((fixed_seq_len,), ragged=True if fixed_seq_len is None else False),
        tf.keras.layers.Masking(mask_value=1),
        tf.keras.layers.Embedding(35000, 400),
        EmbeddingDropout(encoder_dp_rate=0.4, name="emb_dropout"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.SpatialDropout1D(0.3),
        # tf.keras.layers.LSTM(1152, return_sequences=True),
        tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1"),
        tf.keras.layers.SpatialDropout1D(0.5),
        # tf.keras.layers.LSTM(1152, return_sequences=True),
        tf.keras.layers.RNN(AWD_LSTM_Cell2, return_sequences=True, name="AWD_RNN2"),
        tf.keras.layers.SpatialDropout1D(0.5),
        # tf.keras.layers.LSTM(400, return_sequences=True),
        tf.keras.layers.RNN(AWD_LSTM_Cell3, return_sequences=True, name="AWD_RNN3"),
        tf.keras.layers.SpatialDropout1D(0.5),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation='softmax'))
        ])
    if pretrained_weights is not None:
        print("Restoring weights from file... (observe the warnings!)")
        base_model.load_weights(pretrained_weights)
    else:
        print("!!! THE MODEL WEIGHTS ARE UNINITIALIZED !!! You are training the model without embeddings - it *will* overfit.")
    base_model.pop()
    return base_model

def ulmfit_tagger_sequential(*, num_classes=3, pretrained_weights=None, fixed_seq_len=None, spm_model_file,
                             also_return_spm_encoder=False):
    print("Building an AWD-LSTM model using Keras Sequential API")
    # outside_scope_variable = tf.Variable(tf.ones((1152,4608), dtype=tf.float32), trainable=False)
    AWD_LSTM_Cell1 = WeightDropLSTMCell(1152, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    AWD_LSTM_Cell2 = WeightDropLSTMCell(1152, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    AWD_LSTM_Cell3 = WeightDropLSTMCell(400, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    # AWD_LSTM_Cell1 = tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform', dropout=0.5)
    base_model = tf.keras.models.Sequential([
        tf.keras.layers.Input((fixed_seq_len,)),
        tf.keras.layers.Masking(mask_value=1),
        tf.keras.layers.Embedding(35000, 400),
        EmbeddingDropout(encoder_dp_rate=0.4, name="emb_dropout"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.SpatialDropout1D(0.3),
        # tf.keras.layers.LSTM(1152, return_sequences=True),
        tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1"),
        tf.keras.layers.SpatialDropout1D(0.5),
        # tf.keras.layers.LSTM(1152, return_sequences=True),
        tf.keras.layers.RNN(AWD_LSTM_Cell2, return_sequences=True, name="AWD_RNN2"),
        tf.keras.layers.SpatialDropout1D(0.5),
        # tf.keras.layers.LSTM(400, return_sequences=True),
        tf.keras.layers.RNN(AWD_LSTM_Cell3, return_sequences=True, name="AWD_RNN3"),
        tf.keras.layers.SpatialDropout1D(0.5),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation='softmax'))
        ])

    return base_model

def ulmfit_baseline_tagger_functional(*, num_classes=3, pretrained_weights=None, fixed_seq_len=None, spm_model_file,
                          also_return_spm_encoder=False):
    print("Building a regular LSTM model using only standard Keras blocks...")
    AWD_LSTM_Cell1 = WeightDropLSTMCell(1152, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    AWD_LSTM_Cell2 = WeightDropLSTMCell(1152, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    AWD_LSTM_Cell3 = WeightDropLSTMCell(400, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    il = tf.keras.layers.Input((fixed_seq_len,))
    l = tf.keras.layers.Masking(mask_value=1)(il)
    l = tf.keras.layers.Embedding(35000, 400)(l)
    l = EmbeddingDropout(encoder_dp_rate=0.4, name="emb_dropout")(l)
    l = tf.keras.layers.Dropout(0.3)(l)
    l = tf.keras.layers.SpatialDropout1D(0.3)(l)
    #l = tf.keras.layers.LSTM(1152, return_sequences=True)(l)
    l = tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1")(l)
    l = tf.keras.layers.SpatialDropout1D(0.5)(l)
    #l = tf.keras.layers.LSTM(1152, return_sequences=True)(l)
    l = tf.keras.layers.RNN(AWD_LSTM_Cell2, return_sequences=True, name="AWD_RNN2")(l)
    l = tf.keras.layers.SpatialDropout1D(0.5)(l)
    #l = tf.keras.layers.LSTM(400, return_sequences=True)(l)
    l = tf.keras.layers.RNN(AWD_LSTM_Cell3, return_sequences=True, name="AWD_RNN3")(l)
    l = tf.keras.layers.SpatialDropout1D(0.5)(l)
    l = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation='softmax'))(l)
    fake_model = tf.keras.models.Model(inputs=il, outputs=l)
    if pretrained_weights is not None:
        print("Restoring weights from file... (observe the warnings!)")
        fake_model.load_weights(pretrained_weights)
    else:
        print("!!! THE MODEL WEIGHTS ARE UNINITIALIZED !!! You are training the model without embeddings - it *will* overfit.")
    return fake_model
