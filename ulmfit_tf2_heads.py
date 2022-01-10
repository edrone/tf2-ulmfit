import tensorflow as tf
import tensorflow_hub as hub

from ulmfit_tf2 import tf2_ulmfit_encoder, HubRaggedWrapper, TiedDense, RaggedConcatPooler, ConcatPooler


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

    if fixed_seq_len is None:
        il = tf.keras.layers.Input(shape=(None,), ragged=True, name="numericalized_input", dtype=tf.int32)
        kl_restored = HubRaggedWrapper(hub_layer=hub.KerasLayer(restored_hub.signatures['numericalized_encoder'], trainable=True),
                                       name="hub_ulmfit_encoder_ragged")
        kl_tensor = kl_restored(il)
        if return_lm_head:
            if not hasattr(restored_hub, 'lm_head_biases'):
                raise ValueError("This SavedModel was serialized without the LM head biases. Please export "
                                 "from FastAI again.")
            # rt = tf.RaggedTensor.from_row_splits(kl_restored[0], kl_restored[1])
            reference_layer = getattr(restored_hub.encoder_num, 'layer_with_weights-0')
            lm_head_ragged = tf.keras.layers.TimeDistributed(TiedDense(reference_layer, 'softmax'))
            lm_head_ragged.set_weights([restored_hub.lm_head_biases.value()]) # untested
            lm_head_ragged = lm_head_ragged(kl_tensor)
            ret_tensor = lm_head_ragged
            # kl = tf.keras.models.Model(inputs=il, outputs=lm_head_ragged)
            # kl.layers[-1].set_weights()
        else:
            ret_tensor = kl_tensor
    else:
        il = tf.keras.layers.Input(shape=(fixed_seq_len,), name="numericalized_input", dtype=tf.int32)
        kl_tensor = hub.KerasLayer(restored_hub.signatures['numericalized_encoder'], trainable=True, name="ulmfit_encoder")(il)['output']
        if return_lm_head:
            if not hasattr(restored_hub, 'lm_head_biases'):
                raise ValueError("This SavedModel was serialized without the LM head biases. Please export "
                                 "from FastAI again.")
            reference_layer = getattr(restored_hub.encoder_num, 'layer_with_weights-0')
            lm_head = tf.keras.layers.TimeDistributed(TiedDense(reference_layer, 'softmax'))
            lm_head.set_weights([restored_hub.lm_head_biases.value()]) # untested
            lm_head = lm_head(kl_tensor)
            ret_tensor = lm_head
        else:
            ret_tensor = kl_tensor
    model = tf.keras.models.Model(inputs=il, outputs=ret_tensor)
    return model, restored_hub


def ulmfit_sequence_tagger(*, model_type, pretrained_encoder_weights, spm_model_args=None, fixed_seq_len=None, num_classes):

    ######## VERSION 1: ULMFiT sequence tagger model built from Python code - pass the path to a weights directory
    if model_type == 'from_cp':
        ulmfit_rnn_encoder = ulmfit_rnn_encoder_native(pretrained_weights=pretrained_encoder_weights,
                                               spm_model_args=spm_model_args,
                                               fixed_seq_len=fixed_seq_len,
                                               also_return_spm_encoder=False)
        hub_object = None

    ######## VERSION 2: ULMFiT sequence tagged built from a serialized SavedModel - pass the path to a directory containing 'saved_model.pb'
    elif model_type == 'from_hub':
        ulmfit_rnn_encoder, hub_object = ulmfit_rnn_encoder_hub(pretrained_weights=pretrained_encoder_weights,
                                                                spm_model_args=None,
                                                                fixed_seq_len=fixed_seq_len,
                                                                also_return_spm_encoder=False)
    else:
        raise ValueError(f"Unknown model type {args['model_type']}")
    print(f"Adding sequence tagging head with n_classes={num_classes}")
    tagger_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(ulmfit_rnn_encoder.output)
    tagger_model = tf.keras.models.Model(inputs=ulmfit_rnn_encoder.inputs, outputs=tagger_head)
    return tagger_model, hub_object


def ulmfit_last_hidden_state(*, model_type, pretrained_encoder_weights, spm_model_args=None, fixed_seq_len=None):

    ######## VERSION 1: ULMFiT last state built from Python code - pass the path to a weights directory
    if model_type == 'from_cp':
        ulmfit_rnn_encoder = ulmfit_rnn_encoder_native(pretrained_weights=pretrained_encoder_weights,
                                                       spm_model_args=spm_model_args,
                                                       fixed_seq_len=fixed_seq_len,
                                                       also_return_spm_encoder=False)
        hub_object = None

    ######## VERSION 2: ULMFiT last state built from a serialized SavedModel - pass the path to a directory containing 'saved_model.pb'
    elif model_type == 'from_hub':
        ulmfit_rnn_encoder, hub_object = ulmfit_rnn_encoder_hub(pretrained_weights=pretrained_encoder_weights,
                                                                spm_model_args=None,
                                                                fixed_seq_len=fixed_seq_len,
                                                                also_return_spm_encoder=False)
    else:
        raise ValueError(f"Unknown model type {args['model_type']}")
    if fixed_seq_len is None:
        flat_vals = ulmfit_rnn_encoder.output.flat_values
        row_limits = tf.math.subtract(ulmfit_rnn_encoder.output.row_limits(), 1, name="select_last_ragged_idx")
        last_hidden_state = tf.gather(flat_vals, row_limits, name="last_hidden_state_ragged")
    else:
        last_hidden_state = ulmfit_rnn_encoder.output[:, -1, :]
    last_hidden_state_model = tf.keras.models.Model(inputs=ulmfit_rnn_encoder.inputs, outputs=last_hidden_state)
    return last_hidden_state_model, hub_object


def ulmfit_document_classifier(*, model_type, pretrained_encoder_weights, num_classes,
                               spm_model_args=None, fixed_seq_len=None, use_bias=False,
                               with_batch_normalization=False, activation='softmax'):
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
        hub_object=None

    ######## VERSION 2: ULMFiT last state built from a serialized SavedModel - pass the path to a directory containing 'saved_model.pb'
    elif model_type == 'from_hub':
        ulmfit_rnn_encoder, hub_object = ulmfit_rnn_encoder_hub(pretrained_weights=pretrained_encoder_weights,
                                                                spm_model_args=None,
                                                                fixed_seq_len=fixed_seq_len,
                                                                also_return_spm_encoder=False)
    else:
        raise ValueError(f"Unknown model type {args['model_type']}")

    if fixed_seq_len is None:
        rpooler = RaggedConcatPooler(name="RaggedConcatPooler")(ulmfit_rnn_encoder.output)
    else:
        rpooler = ConcatPooler(name="ConcatPooler")(ulmfit_rnn_encoder.output)

    drop_pooler = tf.keras.layers.Dropout(0.2)(rpooler)
    if with_batch_normalization is True:
        bnorm_pooler = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, scale=False, center=False)(drop_pooler)
        bnorm_drop = tf.keras.layers.Dropout(0.1)(bnorm_pooler)
        fc1 = tf.keras.layers.Dense(50, activation='linear', use_bias=use_bias)(bnorm_drop)
        relu1 = tf.keras.layers.ReLU()(fc1)
        bnorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, scale=False, center=False)(relu1)
        drop2 = tf.keras.layers.Dropout(0.1)(bnorm1)
        fc_final = tf.keras.layers.Dense(num_classes, use_bias=use_bias, activation='softmax')(drop2)
    else:
        fc1 = tf.keras.layers.Dense(50, activation='relu', use_bias=use_bias)(drop_pooler)
        drop2 = tf.keras.layers.Dropout(0.1)(fc1)
        fc_final = tf.keras.layers.Dense(num_classes, activation=activation, use_bias=use_bias)(drop2)

    document_classifier_model = tf.keras.models.Model(inputs=ulmfit_rnn_encoder.inputs, outputs=fc_final)
    return document_classifier_model, hub_object


def ulmfit_regressor(*, model_type, pretrained_encoder_weights,
                     spm_model_args=None, fixed_seq_len=None,
                     with_batch_normalization=False):
    """
    Regression head which outputs a single numerical value. The architecture is similar to
    the document classification head and differs only in the last layer (a single neuron
    with no activation instead of softmax).
    """
    regressor, hub_object = ulmfit_document_classifier(model_type=model_type,
                                                       pretrained_encoder_weights=pretrained_encoder_weights,
                                                       num_classes=1,
                                                       spm_model_args=spm_model_args,
                                                       fixed_seq_len=fixed_seq_len,
                                                       use_bias=True,
                                                       with_batch_normalization=with_batch_normalization,
                                                       activation='linear')
    return regressor, hub_object

########### ORIGINAL CODE LEFT FOR REFERENCE ###########

# def ulmfit_tagger_functional(*, num_classes=3, pretrained_weights=None, fixed_seq_len=None):
#     print("Building a regular LSTM model using only standard Keras blocks...")
#     AWD_LSTM_Cell1 = tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform')
#     AWD_LSTM_Cell2 = tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform')
#     AWD_LSTM_Cell3 = tf.keras.layers.LSTMCell(400, kernel_initializer='glorot_uniform')
#     il = tf.keras.layers.Input((fixed_seq_len,), ragged=True if fixed_seq_len is None else False)
#     l = tf.keras.layers.Masking(mask_value=1)(il)
#     l = tf.keras.layers.Embedding(35000, 400)(l)
#     l = EmbeddingDropout(encoder_dp_rate=0.4, name="emb_dropout")(l)
#     l = tf.keras.layers.Dropout(0.3)(l)
#     l = tf.keras.layers.SpatialDropout1D(0.3)(l)
#     l = tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1")(l)
#     l = tf.keras.layers.SpatialDropout1D(0.5)(l)
#     l = tf.keras.layers.RNN(AWD_LSTM_Cell2, return_sequences=True, name="AWD_RNN2")(l)
#     l = tf.keras.layers.SpatialDropout1D(0.5)(l)
#     l = tf.keras.layers.RNN(AWD_LSTM_Cell3, return_sequences=True, name="AWD_RNN3")(l)
#     l = tf.keras.layers.SpatialDropout1D(0.5)(l)
#     l = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(l)
#     fake_model = tf.keras.models.Model(inputs=il, outputs=l)
#     if pretrained_weights is not None:
#         print("Restoring weights from file... (observe the warnings!)")
#         fake_model.load_weights(pretrained_weights)
#     else:
#         print("!!! THE MODEL WEIGHTS ARE UNINITIALIZED !!!")
#     return fake_model
