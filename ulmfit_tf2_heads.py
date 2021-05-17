import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.keras.engine import keras_tensor
# from .awdlstm_tf2 import *
from .ulmfit_tf2 import *

def ulmfit_rnn_encoder_native(*, pretrained_weights=None, fixed_seq_len=None, spm_model_args,
                              also_return_spm_encoder=False, return_lm_head=False):
    """ Returns an ULMFiT encoder from Python code """
    print("Building model from Python code (not tf.saved_model)...")
    lm_num, enc_num, _, spm_encoder_model = tf2_ulmfit_encoder(fixed_seq_len=fixed_seq_len, spm_args=spm_model_args,
                                                               flatten_ragged_outputs=False)
    if pretrained_weights is not None:
        print("Restoring weights from file....")
        lm_num.load_weights(pretrained_weights)
    else:
        print("!!! THE MODEL WEIGHTS ARE UNINITIALIZED !!! Make sure to restore them from file.")
    ret_layer = lm_num if return_lm_head is True else enc_num
    if also_return_spm_encoder is True:
        return ret_layer, spm_encoder_model
    else:
        return ret_layer

def ulmfit_rnn_encoder_hub(*, pretrained_weights=None, fixed_seq_len=None, spm_model_args=None,
                              also_return_spm_encoder=False, return_lm_head=False):
    """ Returns an ULMFiT encoder from a serialized SavedModel  """
    if also_return_spm_encoder:
        print(f"Info: The SPM layer is baked into the SavedModel. It will not be returned separately.")
    if spm_model_args is not None:
        print(f"Info: When restoring the ULMFiT encoder from a SavedModel, `spm_model_args` has no effect`")
    restored_hub = hub.load(pretrained_weights)

    if fixed_seq_len == None: # TODO: check tensorspec and print a pretty info if ragged parameters here and in the SavedModel are incompatible
        il = tf.keras.layers.Input(shape=(None,), ragged=True, name="numericalized_input", dtype=tf.int32)
        #il_rows = tf.keras.layers.Input(shape=(None,), name="rowsplits", dtype=tf.int64)
        # kl = hub.KerasLayer(restored_hub.signatures['numericalized_encoder'], trainable=True, name="ulmfit_encoder")
        kl_restored = hub.KerasLayer(restored_hub.encoder_num, trainable=True)(il)
        if return_lm_head:
            if not hasattr(restored_hub, 'lm_head_biases'):
                raise ValueError("This SavedModel was serialized without the LM head biases. Please export from FastAI again.")
            rt = tf.RaggedTensor.from_row_splits(kl_restored[0], kl_restored[1])
            reference_layer = getattr(restored_hub.encoder_num, 'layer_with_weights-0')
            lm_head_ragged = tf.keras.layers.TimeDistributed(TiedDense(reference_layer, 'softmax'))(rt)
            kl = tf.keras.models.Model(inputs=il, outputs=lm_head_ragged)
            kl.layers[-1].set_weights([restored_hub.lm_head_biases.value()])
        else:
            kl = kl_restored
        # model = tf.keras.models.Model(inputs=il, outputs=rec_ragged_tensor)
    else:
        il = tf.keras.layers.Input(shape=(fixed_seq_len,), name="numericalized_input", dtype=tf.int32)
        kl_restored = hub.KerasLayer(restored_hub.signatures['numericalized_encoder'], trainable=True, name="ulmfit_encoder")(il)['output']
        if return_lm_head:
            if not hasattr(restored_hub, 'lm_head_biases'):
                raise ValueError("This SavedModel was serialized without the LM head biases. Please export from FastAI again.")
            reference_layer = getattr(restored_hub.encoder_num, 'layer_with_weights-0')
            lm_head = tf.keras.layers.TimeDistributed(TiedDense(reference_layer, 'softmax'))(kl_restored)
            kl = tf.keras.models.Model(inputs=il, outputs=lm_head)
            kl.layers[-1].set_weights([restored_hub.lm_head_biases.value()])
        else:
            kl = kl_restored
    return il, kl, restored_hub

def ulmfit_sequence_tagger_head(*, enc_num, model_type='from_cp', num_classes=3, fixed_seq_len=None, **kwargs):
    """ Convenience method to put a sequence-tagging head on top of the ULMFiT encoder.
    
        `enc_num`       - the ULMFiT encoder (can be None if using a SavedModel)
        `model_type`    - 'from_cp' if encoder is built from Python code or `from_hub` if it's restored from a SavedModel
        `num_classes`   - how many classes there are
        `fixed_seq_len` - (for SavedModel only) Fixed sequence length to which all training examples will be padded
                                                if necessary. `None` means the SavedModel will use RaggedTensors and
                                                variable sequence length. You must set this parameter exactly in the
                                                same way as you did when exporting the encoder.
        `kwargs`        - (for SavedModel only) contains two keras objects:
                          'input_layer' - an instance of tf.keras.layers.Input which can accept ragged
                                          or fixed-length inputs, depending on how the encoder was exported.
                          'keras_layer' - the encoder output either as a fixed-sequence length batch, or
                                          an array of [flat_values, row_splits] restored back into a RaggedTensor.
    """
    if model_type == 'from_hub':
        print(f"Adding sequence tagging head with n_classes={num_classes} (TF Hub)")
        if fixed_seq_len is None:
            il = kwargs['input_layer']
            kl = kwargs['keras_layer']
            ragged_restored = tf.RaggedTensor.from_row_splits(kl[0], kl[1])
            tagger_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes))(ragged_restored)
            tagger_model = tf.keras.models.Model(inputs=il, outputs=tagger_head)
        else:
            il = kwargs['input_layer']
            kl = kwargs['keras_layer']
            tagger_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes))(kl)
            tagger_model = tf.keras.models.Model(inputs=il, outputs=tagger_head)
    elif model_type == 'from_cp':
        print(f"Adding sequence tagging head with n_classes={num_classes}")
        tagger_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(enc_num.output)
        tagger_model = tf.keras.models.Model(inputs=enc_num.inputs, outputs=tagger_head)
    else:
        raise ValueError(f"Unknown model source {model_type}")
    return tagger_model

################################### end to end methods ###############################
def ulmfit_sequence_tagger(*, model_type, pretrained_encoder_weights, spm_model_args=None, fixed_seq_len=None, num_classes=3):

    ######## VERSION 1: ULMFiT sequence tagger model built from Python code - pass the path to a weights directory
    if model_type == 'from_cp':
        ulmfit_rnn_encoder = ulmfit_rnn_encoder_native(pretrained_weights=pretrained_encoder_weights,
                                               spm_model_args=spm_model_args,
                                               fixed_seq_len=fixed_seq_len,
                                               also_return_spm_encoder=False)
        hub_object = il = kl = None

    ######## VERSION 2: ULMFiT sequence tagged built from a serialized SavedModel - pass the path to a directory containing 'saved_model.pb'
    elif model_type == 'from_hub':
        il, kl, hub_object = ulmfit_rnn_encoder_hub(pretrained_weights=pretrained_encoder_weights,
                                                     spm_model_args=None,
                                                     fixed_seq_len=fixed_seq_len,
                                                     also_return_spm_encoder=False)
        ulmfit_rnn_encoder = None
    else:
        raise ValueError(f"Unknown model type {args['model_type']}")
    ulmfit_tagger = ulmfit_sequence_tagger_head(enc_num=ulmfit_rnn_encoder,
                                                model_type=model_type,
                                                num_classes=num_classes,
                                                fixed_seq_len=fixed_seq_len,
                                                input_layer=il,
                                                keras_layer=kl)
    if ulmfit_rnn_encoder is not None:
        ulmfit_rnn_encoder.summary()
    return ulmfit_tagger, hub_object

def ulmfit_last_hidden_state(*, model_type, pretrained_encoder_weights, spm_model_args=None, fixed_seq_len=None):

    ######## VERSION 1: ULMFiT last state built from Python code - pass the path to a weights directory
    if model_type == 'from_cp':
        ulmfit_rnn_encoder = ulmfit_rnn_encoder_native(pretrained_weights=pretrained_encoder_weights,
                                               spm_model_args=spm_model_args,
                                               fixed_seq_len=fixed_seq_len,
                                               also_return_spm_encoder=False)
        if fixed_seq_len is None:
            flat_vals = ulmfit_rnn_encoder.output.flat_values
            row_limits = tf.math.subtract(ulmfit_rnn_encoder.output.row_limits(), 1, name="select_last_ragged_idx")
            last_hidden_state = tf.gather(flat_vals, row_limits, name="last_hidden_state_ragged")
        else:
            last_hidden_state = ulmfit_rnn_encoder.output[:, -1, :]
        last_hidden_state_model = tf.keras.models.Model(inputs=ulmfit_rnn_encoder.inputs, outputs=last_hidden_state)

    ######## VERSION 2: ULMFiT last state built from a serialized SavedModel - pass the path to a directory containing 'saved_model.pb'
    elif model_type == 'from_hub':
        il, kl, hub_object = ulmfit_rnn_encoder_hub(pretrained_weights=pretrained_encoder_weights,
                                                     spm_model_args=None,
                                                     fixed_seq_len=fixed_seq_len,
                                                     also_return_spm_encoder=False)
        if fixed_seq_len is None:
            row_limits = kl[1][1:] - 1 # equivalent to row_splits - get the last index in the ragged dimension
            last_hidden_state = tf.gather(kl[0], row_limits, name="last_hidden_state_ragged")
        else:
            last_hidden_state = kl.output[:, -1, :]
        last_hidden_state_model = tf.keras.models.Model(inputs=il, outputs=last_hidden_state)
    else:
        raise ValueError(f"Unknown model type {args['model_type']}")
    return last_hidden_state_model

def ulmfit_document_classifier(*, model_type, pretrained_encoder_weights, num_classes, spm_model_args=None, fixed_seq_len=None):
    """
    Document classification head as per the ULMFiT paper:
       - AvgPool + MaxPool + Last hidden state
       - BatchNorm
       - 2 FC layers
    """
    ######## VERSION 1: ULMFiT last state built from Python code - pass the path to a weights directory
    if model_type == 'from_cp':
        ulmfit_rnn_encoder = ulmfit_rnn_encoder_native(pretrained_weights=pretrained_encoder_weights,
                                               spm_model_args=spm_model_args,
                                               fixed_seq_len=fixed_seq_len,
                                               also_return_spm_encoder=False)
        if fixed_seq_len is None:
            rpooler = RaggedConcatPooler(name="RaggedConcatPooler")(ulmfit_rnn_encoder.output)
        else:
            rpooler = ConcatPooler(name="ConcatPooler")(ulmfit_rnn_encoder.output)
        hub_object=None

    ######## VERSION 2: ULMFiT last state built from a serialized SavedModel - pass the path to a directory containing 'saved_model.pb'
    elif model_type == 'from_hub':
        il, kl, hub_object = ulmfit_rnn_encoder_hub(pretrained_weights=pretrained_encoder_weights,
                                                     spm_model_args=None,
                                                     fixed_seq_len=fixed_seq_len,
                                                     also_return_spm_encoder=False)
        if fixed_seq_len is None:
            rpooler = RaggedConcatPooler(inputs_are_flattened=True, name="RaggedConcatPooler")(kl)
        else:
            rpooler = ConcatPooler(name="ConcatPooler")(kl)
    else:
        raise ValueError(f"Unknown model type {args['model_type']}")
    bnorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(rpooler)
    drop1 = tf.keras.layers.Dropout(0.4)(bnorm1)
    fc1 = tf.keras.layers.Dense(50, activation='relu')(drop1)
    bnorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(fc1)
    drop2 = tf.keras.layers.Dropout(0.1)(bnorm2)
    fc_final = tf.keras.layers.Dense(num_classes, activation='softmax')(drop2)
    if model_type == 'from_cp':
        document_classifier_model = tf.keras.models.Model(inputs=ulmfit_rnn_encoder.input, outputs=fc_final)
    elif model_type == 'from_hub':
        document_classifier_model = tf.keras.models.Model(inputs=il, outputs=fc_final)
    else:
        raise ValueError(f"Unknown model type {args['model_type']}")
    return document_classifier_model, hub_object

########### ## THE CODE BELOW THIS LINE IS FOR DEBUGGING / UNSTABLE / EXPERIMENTAL ###########

def ulmfit_tagger_functional(*, num_classes=3, pretrained_weights=None, fixed_seq_len=None):
    print("Building a regular LSTM model using only standard Keras blocks...")
    AWD_LSTM_Cell1 = tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform')
    AWD_LSTM_Cell2 = tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform')
    AWD_LSTM_Cell3 = tf.keras.layers.LSTMCell(400, kernel_initializer='glorot_uniform')
    il = tf.keras.layers.Input((fixed_seq_len,), ragged=True if fixed_seq_len is None else False)
    l = tf.keras.layers.Masking(mask_value=1)(il)
    l = tf.keras.layers.Embedding(35000, 400)(l)
    l = EmbeddingDropout(encoder_dp_rate=0.4, name="emb_dropout")(l)
    l = tf.keras.layers.Dropout(0.3)(l)
    l = tf.keras.layers.SpatialDropout1D(0.3)(l)
    l = tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1")(l)
    l = tf.keras.layers.SpatialDropout1D(0.5)(l)
    l = tf.keras.layers.RNN(AWD_LSTM_Cell2, return_sequences=True, name="AWD_RNN2")(l)
    l = tf.keras.layers.SpatialDropout1D(0.5)(l)
    l = tf.keras.layers.RNN(AWD_LSTM_Cell3, return_sequences=True, name="AWD_RNN3")(l)
    l = tf.keras.layers.SpatialDropout1D(0.5)(l)
    l = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(l)
    fake_model = tf.keras.models.Model(inputs=il, outputs=l)
    if pretrained_weights is not None:
        print("Restoring weights from file... (observe the warnings!)")
        fake_model.load_weights(pretrained_weights)
    else:
        print("!!! THE MODEL WEIGHTS ARE UNINITIALIZED !!!")
    return fake_model
