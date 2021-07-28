import math

import matplotlib.pyplot as plt  # for LRFinder
import numpy as np
import tensorflow as tf
import tensorflow_text as text
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tqdm import trange


def tf2_ulmfit_encoder(*, fixed_seq_len=None, flatten_ragged_outputs=True, spm_args=None, vocab_size=None):
    """ Builds an ULMFiT as a model trainable in Keras.

        :param fixed_seq_len:            if set to `None`, builds a variable-length model with RaggedTensors.
                                         Otherwise uses fixed-length sequences with 1 (not zero!) as padding.
        :param flatten_ragged_outputs:   if set to `True`, the RaggedTensor output will be returned in a "decomposed"
                                         representation of flat_values and row_splits (see RaggedTensor documentation).
                                         You probably want to set this to `True` if preparing a model for serialization
                                         and `False` if your code is entirely in Python and you only save Keras weights.
        :param spm_args:                 a dictionary containing configuration for the SPMNumericalizer layer.
                                         Valid keys are: `spm_model_file`, `add_bos`, `add_eos`, `name` and
                                         `lumped_sents_separator`. If you pass `None`, the tokenizer and numericalizer
                                         will not be created and you will need to numericalize the data yourself.
        :param vocab_size:               (only relevant if `spm_args` is None) - number of subwords in the
                                         vocabulary.
        :return: Returns four instances of tf.keras.Model:
        lm_model_num - encoder with a language modelling head on top (and weights tied to embeddings).
                       This version accepts already numericalized text.
                       * Example call (fixed length):

                       dziendobry = tf.constant([[11406,  7465, 34951,   218, 34992, 34967, 12545, 34986] + [1]*92])
                       lm_num(dziendobry)

                       Note that the final 92 padding tokens are masked throughout the model - this is taken care of
                       by the `compute_output_mask` in successive layers.

                       * Example call (variable length):

                       dziendobry = tf.ragged.constant([[11406,  7465, 34951,   218, 34992, 34967, 12545, 34986]])
                       lm_num(dziendobry)

        encoder_num - returns only the outputs of the last RNN layer (dim 400 as per ULMFiT paper). Accepts
                      already numericalized text. Calling convention is same as for lm_model_num.
                      Again, note the presence of _keras_mask in the output on padding tokens.

        outmask_num - returns explicit mask for an input sequence. Not used in the model itself, but might be useful
                      for working with some signatures in the serialized version.

  spm_encoder_model - if `spm_args` was passed, this holds the numericalizer (otherwise None is returned).
                      The numericalizer accepts a string and outputs its sentencepiece representation. The SPM model
                      must be trained externally and a path needs to be provided in `spm_args{'spm_model_file'}`
                      (it is also serialized as a tf.saved_model.Asset). In a fixed length setting,
                      this layer TRUNCATES the text if it's longer than fixed_seq_len tokens and AUTOMATICALLY
                      ADDS PADDING WITH A VALUE OF 1 if needed.
    """

    ##### STAGE 1 - BUILD AN SPM ENCODER LAYER #####
    if not any([spm_args, vocab_size]):
        raise ValueError("Please either explicitly provide the vocabulary size or a path to an SPM model file.")
    uniform_initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None) # initializer for embeddings

    if spm_args is not None: # do not attach a string numericalizer if spm_args isn't passed
        seq_type = "ragged" if fixed_seq_len is None else "fixed"
        string_input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name=f"{seq_type}_string_input")
        spm_layer = SPMNumericalizer(spm_path=spm_args['spm_model_file'],
                                     add_bos=spm_args.get('add_bos') or False,
                                     add_eos=spm_args.get('add_eos') or False,
                                     fixed_seq_len=fixed_seq_len,
                                     lumped_sents_separator=spm_args.get('lumped_sents_separator') or "",
                                     name=f"{seq_type}_spm_numericalizer")
        numericalized_layer = spm_layer(string_input_layer)
        vocab_size_ = spm_layer.spmproc.vocab_size().numpy()
        spm_encoder_model = tf.keras.Model(inputs=string_input_layer, outputs=numericalized_layer)
    else:
        vocab_size_ = vocab_size
        spm_encoder_model = None

    ##### STAGE 2 - SET UP EMBEDDINGS #####
    # Unfortunately not all Keras objects serialize well when using RaggedTensors.
    # In such cases we provide serializable wrappers around the unruly layers.
    if fixed_seq_len is None:
        print(f"Building an ULMFiT model with: \n1) a variable sequence and RaggedTensors"
              f"2) a vocabulary size of {vocab_size}.")
        print("=====================================================================================================")
        print("NOTE: THIS MODEL USES RAGGED TENSORS AND IS SERIALIZABLE TO A SavedModel WITH A WORKAROUND.")
        print("The serialized version will NOT accept or output RaggedTensors. This is a TensorFlow issue")
        print("which may be resolved in the future. In the meantime, you need to convert your data from and to")
        print("`flat_values` and `row_splits` as follows:\n")
        print("input_numericalized = tf.ragged.constant([[30, 40, 50, 110], [20, 30]])")
        print("flatvals = input_numericalized.flat_values")
        print("rowspl = input_numericalized.row_splits")
        print("ret = hub_object.signatures['numericalized_encoder'](flatvals=flatvals, rowspl=rowspl")
        print("ret = tf.RaggedTensor.from_row_splits(ret['output_flat'], ret['output_rows']) \n")
        print("======================================================================================-==============")
        EmbedDropLayer = RaggedEmbeddingDropout
        SpatialDrop1DLayer = RaggedSpatialDropout1D
        layer_name_prefix = "ragged_"
    else:
        print(f"Building an ULMFiT model with: \n1) a fixed sequence length of {fixed_seq_len}\n"
              f"2) a vocabulary size of {vocab_size}.")
        EmbedDropLayer = EmbeddingDropout
        SpatialDrop1DLayer = tf.keras.layers.SpatialDropout1D
        layer_name_prefix = ""

    embedz = CustomMaskableEmbedding(vocab_size_, 400, embeddings_initializer=uniform_initializer,
                                     mask_zero=False,
                                     mask_value=None if fixed_seq_len is None else 1,
                                     name="ulmfit_embeds")
    encoder_dropout = EmbedDropLayer(encoder_dp_rate=0.4, name=f"{layer_name_prefix}emb_dropout")
    input_dropout = SpatialDrop1DLayer(0.4, name=f"{layer_name_prefix}inp_dropout")

    ###### STAGE 3 - RECURRENT LAYERS ######
    # Plain LSTM cells - we will apply AWD manually in the training loop
    # It turns out that the generic RNN API below will not use CuDNN kernel for LSTM networks.
    #rnn1 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform'),
    #                           return_sequences=True, name="AWD_RNN1")
    #rnn1_drop = SpatialDrop1DLayer(0.3, name=f"{layer_name_prefix}rnn_drop1")
    #rnn2 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(1152, kernel_initializer='glorot_uniform'),
    #                           return_sequences=True, name="AWD_RNN2")
    #rnn2_drop = SpatialDrop1DLayer(0.3, name=f"{layer_name_prefix}rnn_drop2")
    #rnn3 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(400, kernel_initializer='glorot_uniform'),
    #                           return_sequences=True, name="AWD_RNN3")
    #rnn3_drop = SpatialDrop1DLayer(0.4, name=f"{layer_name_prefix}rnn_drop3")


    # However, invoking the LSTM layer directly uses CuDNN if available
    rnn1 = tf.keras.layers.LSTM(1152, kernel_initializer='glorot_uniform', return_sequences=True,
                                name=f"AWD_RNN1")
    rnn1_drop = SpatialDrop1DLayer(0.3, name=f"rnn_drop1")
    rnn2 = tf.keras.layers.LSTM(1152, kernel_initializer='glorot_uniform', return_sequences=True,
                                name=f"AWD_RNN2")
    rnn2_drop = SpatialDrop1DLayer(0.3, name=f"rnn_drop2")
    rnn3 = tf.keras.layers.LSTM(400, kernel_initializer='glorot_uniform', return_sequences=True,
                                 name=f"AWD_RNN3")
    rnn3_drop = SpatialDrop1DLayer(0.2, name=f"rnn_drop3")

    ###### STAGE 4. THE ACTUAL ENCODER MODEL ######
    numericalized_input = tf.keras.layers.Input(shape=(fixed_seq_len,), dtype=tf.int32,
                                                name=f"{layer_name_prefix}numericalized_input",
                                                ragged=True if fixed_seq_len is None else False)
    explicit_mask = ExplicitMaskGenerator(mask_value=1)(numericalized_input)
    m = embedz(numericalized_input)
    m = encoder_dropout(m)
    m = input_dropout(m)
    m = rnn1(m)
    m = rnn1_drop(m)
    m = rnn2(m)
    m = rnn2_drop(m)
    m = rnn3(m)
    rnn_encoder = rnn3_drop(m)

    ###### OPTIONAL LANGUAGE MODELLING HEAD FOR FINETUNING #######
    fc_head = tf.keras.layers.TimeDistributed(TiedDense(reference_layer=embedz, activation='softmax'),
                                              name='lm_head_tied')
    fc_head_dp = tf.keras.layers.Dropout(0.05)
    lm = fc_head(rnn_encoder)
    lm = fc_head_dp(lm)

    ##### ALL MODELS ASSEMBLED TOGETHER #####
    lm_model_num = tf.keras.Model(inputs=numericalized_input, outputs=lm)
    if fixed_seq_len is None and flatten_ragged_outputs is True:
        # RaggedTensors as outputs are not serializable when using signatures. This *may* be fixed in TF 2.5.1
        encoder_num = tf.keras.Model(inputs=numericalized_input,
                                     outputs=[rnn_encoder.flat_values, rnn_encoder.row_splits])
    else:
        encoder_num = tf.keras.Model(inputs=numericalized_input, outputs=rnn_encoder)
    outmask_num = tf.keras.Model(inputs=numericalized_input, outputs=explicit_mask)
    return lm_model_num, encoder_num, outmask_num, spm_encoder_model

class ExportableULMFiT(tf.keras.Model):
    """
    This class encapsulates a TF2 SavedModel serializable version of ULMFiT with a couple of useful
    signatures for flexibility.

    Serialization procedure:

    spm_args = {'spm_model_file': '/tmp/plwiki100-sp35k.model', 'add_bos': True, 'add_eos': True,
                'lumped_sents_separator': '[SEP]'}
    lm_num, enc_num, outmask_num, spm_encoder_model = tf2_ulmfit_encoder(fixed_seq_len=200, spm_args=spm_args)
    tf.keras.backend.set_learning_phase(0)
    exportable = ExportableULMFiT(encoder_num, outmask_num, spm_encoder_model)
    convenience_signatures={'numericalized_encoder': exportable.numericalized_encoder,
                            'string_encoder': exportable.string_encoder,
                            'spm_processor': exportable.string_numericalizer}
    tf.saved_model.save(exportable, 'ulmfit_tf2', signatures=convenience_signatures)


    Deserialization (you don't need any Python code and it really works with all the custom tweaks!):

    import tensorflow_text # must do it explicitly!
    import tensorflow_hub as hub
    import tensorflow as tf

    restored_hub = hub.load('ulmfit_tf2')   # now you can work with functions listed in signatures:
    hello_vectors = restored_hub(tf.constant(["Dzień dobry, ULMFiT!"]))

    Note the above examples return dictionaries with the encoder, numericalized tokens and mask outputs.
    If you want to use RNN vectors as a Keras layer you can access the serialized model
    directly like this:

    rnn_encoder = hub.KerasLayer(restored_hub.encoder_str, trainable=True) # or .encoder_num for numericalized inputs
    hello_vectors = rnn_encoder(tf.constant(['Dzień dobry, ULMFiT']))

    If you want, you can also manually verify that all the fancy dropouts from the ULMFiT paper are there:

    tf.keras.backend.set_learning_phase(1)
    Now call `rnn_encoder(tf.constant([['Dzień dobry, ULMFiT']]))` a couple of times - you will see
    values changing all the time (due to WeightDrop in the RNN layers) and some zeros (due to regular
    dropout on the output).
    """

    def __init__(self, encoder_num, outmask_num, spm_encoder_model, lm_head_biases=None):
        super().__init__()
        self.encoder_num = encoder_num
        self.masker_num = outmask_num
        self.spm_encoder_model = spm_encoder_model
        self.lm_head_biases = tf.Variable(initial_value=lm_head_biases) if lm_head_biases is not None else None
        self.encoder_str = tf.keras.Model(inputs=self.spm_encoder_model.inputs,
                                          outputs=self.encoder_num(self.spm_encoder_model.outputs))

    @tf.function(input_signature=[tf.TensorSpec((None,), dtype=tf.string)])
    def __call__(self, x):
        tf.print("WARNING: to obtain a trainable model, please wrap the `string_encoder` " \
                 "or `numericalized_encoder` signature into a hub.KerasLayer(..., trainable=True) object. \n")
        return self.string_encoder(x)

    @tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.int32)])
    def numericalized_encoder(self, numericalized):
        mask = self.masker_num(numericalized)
        return {'output': self.encoder_num(numericalized),
                'mask': mask}

    @tf.function(input_signature=[tf.TensorSpec((), dtype=tf.float32)])
    def apply_awd(self, awd_rate):
        # tf.print("Applying AWD in graph mode")
        rnn1_w = self.encoder_num.get_layer("AWD_RNN1").variables
        rnn2_w = self.encoder_num.get_layer("AWD_RNN2").variables
        rnn3_w = self.encoder_num.get_layer("AWD_RNN3").variables

        w1_mask = tf.nn.dropout(tf.fill(rnn1_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn1_w[1].assign(w1_mask * rnn1_w[1])

        w2_mask = tf.nn.dropout(tf.fill(rnn2_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn2_w[1].assign(w2_mask * rnn2_w[2])

        w3_mask = tf.nn.dropout(tf.fill(rnn3_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn3_w[1].assign(w3_mask * rnn3_w[2])

    @tf.function(input_signature=[tf.TensorSpec([None, ], dtype=tf.string)])
    def string_encoder(self, string_inputs):
        numerical_representation = self.string_numericalizer(string_inputs)
        hidden_states = self.numericalized_encoder(numerical_representation['numericalized'])['output']
        return {'output': hidden_states,
                'numericalized': numerical_representation['numericalized'],
                'mask': numerical_representation['mask']}

    @tf.function(input_signature=[tf.TensorSpec([None, ], dtype=tf.string)])
    def string_numericalizer(self, string_inputs):
        numerical_representation = self.spm_encoder_model(string_inputs)
        mask = self.masker_num(numerical_representation)
        return {'numericalized': numerical_representation,
                'mask': mask}

    ################## UNSUPPORTED / EXPERIMENTAL #################

    #@tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.int32)])
    #def numericalized_lm_head(self, numericalized):
    #    # return {'lm_head': self.lm_model(numericalized)}
    #    return self.lm_model_num(numericalized)
    #
    #@tf.function(input_signature=[tf.TensorSpec((None,), dtype=tf.string)])
    #def string_lm_head(self, string_inputs):
    #    return self.lm_model_str(string_inputs)

class ExportableULMFiTRagged(tf.keras.Model):
    """ Same as ExportableULMFiT but supports RaggedTensors with a workaround """
    def __init__(self, encoder_num, outmask_num, spm_encoder_model, lm_head_biases=None, scheduler=None):
        super().__init__()
        self.encoder_num = encoder_num
        self.masker_num = outmask_num
        self.spm_encoder_model = spm_encoder_model
        self.lm_head_biases = tf.Variable(initial_value=lm_head_biases) if lm_head_biases is not None else None
        self.stlr_scheduler = scheduler

    # def __call__(self, x):
    #     rag_num = self.string_numericalizer(x)['numericalized']
    #     return self.numericalized_encoder(rag_num)

    # Calling this signature from a hub.KerasLayer wrapper gives errors - since TF 2.4.1
    # tf.keras.layers.Input produces a KerasTensor, which is not compatible with tf.Tensor.
    # I found the only way to pass named parameters `flatvals` and `rowspl` is to
    # wrap hub.KerasLayer around HubRaggedWrapper. Yes, that's a wrapper around a wrapper...
    @tf.function(input_signature=[tf.TensorSpec([None, ], dtype=tf.int32),
                                  tf.TensorSpec([None, ], dtype=tf.int64)])
    def numericalized_encoder(self, flatvals, rowspl):
        ret = self.encoder_num(tf.RaggedTensor.from_row_splits(flatvals, rowspl))
        return {'output_flat': ret[0],
                'output_rows': ret[1]}

    @tf.function(input_signature=[tf.TensorSpec([None, ], dtype=tf.string)])
    def string_encoder(self, string_inputs):
        numerical_representation = self.spm_encoder_model(string_inputs)
        ret = self.encoder_num(numerical_representation)
        return {'output_flat': ret[0],
                'output_rows': ret[1]}

    @tf.function(input_signature=[tf.TensorSpec([None, ], dtype=tf.string)])
    def string_numericalizer(self, string_inputs):
        # string_inputs = tf.expand_dims(string_inputs, axis=-1)
        numerical_representation = self.spm_encoder_model(string_inputs)
        mask = self.masker_num(numerical_representation)
        return {'numericalized_flat': numerical_representation.flat_values,
                'numericalized_rows': numerical_representation.row_splits}

    @tf.function(input_signature=[tf.TensorSpec((), dtype=tf.float32)])
    def apply_awd(self, awd_rate):
        # tf.print("Applying AWD in graph mode and ragged tensors")
        rnn1_w = self.encoder_num.get_layer("AWD_RNN1").variables
        rnn2_w = self.encoder_num.get_layer("AWD_RNN2").variables
        rnn3_w = self.encoder_num.get_layer("AWD_RNN3").variables

        w1_mask = tf.nn.dropout(tf.fill(rnn1_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn1_w[1].assign(w1_mask * rnn1_w[1])

        w2_mask = tf.nn.dropout(tf.fill(rnn2_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn2_w[1].assign(w2_mask * rnn2_w[2])

        w3_mask = tf.nn.dropout(tf.fill(rnn3_w[1].shape, 1-awd_rate), rate=awd_rate)
        rnn3_w[1].assign(w3_mask * rnn3_w[2])


def keras_register_once(package='Custom', name=None):
    """ A decorator that registers the wrapped class in Keras serialization framework only once """
    if name is None:
        raise ValueError(f"Please provide a name for the serializable Keras object")
    def decorator(cls):
        if tf.keras.utils.get_registered_object(f"{package}>{name}") is None:
            print(f"Registering a Keras serializable object `{package}>{name}`")
            registry = tf.keras.utils.register_keras_serializable(package=package, name=name)
            registry(cls)
        else:
            print(f"Keras serializable object `{package}>{name}` already registered")
        return cls
    return decorator

# @tf.keras.utils.register_keras_serializable()
@keras_register_once(package='Custom', name='SPMNumericalizer')
class SPMNumericalizer(tf.keras.layers.Layer):
    """ A serializable Keras layer which wraps the text.SentencepieceTokenizer object

        Notice that the model file will be conveniently saved in the 'assets' directory.
    """
    def __init__(self, name=None, spm_path=None, fixed_seq_len=None,
                 pad_value=1, add_bos=False, add_eos=False, lumped_sents_separator="", **kwargs):
        self.spm_path = spm_path
        self.add_bos = add_bos
        self.add_eos = add_eos
        if isinstance(spm_path, tf.saved_model.Asset):
            self.spm_asset = spm_path
        else:
            self.spm_asset = tf.saved_model.Asset(self.spm_path)
        self.spm_proto = tf.io.read_file(self.spm_asset).numpy()
        self.spmproc = text.SentencepieceTokenizer(self.spm_proto, add_bos=self.add_bos, add_eos=self.add_eos)
        self.fixed_seq_len = fixed_seq_len
        self.pad_value = pad_value
        self.lumped_sents_separator = lumped_sents_separator
        super().__init__(name=name, **kwargs)
        self.trainable = False

    def build(self, input_shape):
        print(f">>>> INSIDE BUILD / SPMTOK <<<< {input_shape} ")
        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=None):
        if tf.not_equal(self.lumped_sents_separator, ""):
            splitted = tf.strings.split(inputs, self.lumped_sents_separator)
            ret = self.spmproc.tokenize(splitted)
            ret = ret.merge_dims(1, 2)
            #ret = tf.strings.join(ret)
        else:
            ret = self.spmproc.tokenize(inputs)
        if self.fixed_seq_len is not None:
            ret_padded = ret.to_tensor(self.pad_value)
            #ret_padded = tf.squeeze(ret_padded, axis=1)
            ret_padded = tf.pad(ret_padded, tf.constant([[0, 0, ], [0, self.fixed_seq_len, ]]),
                                constant_values=self.pad_value)
            ret_padded = ret_padded[:, :self.fixed_seq_len]
            return ret_padded
        else:
            # ret = tf.squeeze(ret, axis=1)
            return ret

    # @tf.function(input_signature=[tf.TensorSpec((), dtype=tf.string)])
    def set_sentence_separator(self, sep):
        """ Insert additional <s> and </s> tokens between sentences in a single training example.

            This can be useful if working with short documents on which
            for some reason the model performs better if they are sentence-tokenized.
            
            The `sep` symbol is a separator by which each input example is split and
            surrounded by <s>...</s> tokens (only if `add_bos` and `add_eos` options
            for the SPMNumericalizer were set to True). For example, this input:

            The cat sat on a mat. [SEP] And spat.

            can be encoded as:
            <s> The cat sat on a mat. </s> <s> And spat. </s>
        """
        self.lumped_sents_separator = sep

    def compute_output_shape(self, input_shape):
        tf.print(f"INPUT SHAPE IS {input_shape}")
        if self.fixed_seq_len is None:
            # return tf.TensorShape(input_shape[0], None)
            return (input_shape[0], None)
        else:
            # return tf.TensorShape([input_shape[0], self.fixed_seq_len])
            return (input_shape[0], self.fixed_seq_len)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'spm_path': self.spm_path,
                    'fixed_seq_len': self.fixed_seq_len,
                    'pad_value': self.pad_value,
                    'add_bos': self.add_bos,
                    'add_eos': self.add_eos,
                    'lumped_sents_separator': ""})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class HubRaggedWrapper(tf.keras.layers.Layer):
    """ A workaround for a possible bug in TF SavedModel signatures

     TF Hub API doesn't allow you to call hub_obj.signatures["serving_default"](**tensors_in)
     on ragged tensors. However, it turns out we can call the concrete function directly via the
     `resolved_object` attribute and pass the **tensors_in kwargs there directly.

    """
    def __init__(self, *, hub_layer, **kwargs):
        super(HubRaggedWrapper, self).__init__(**kwargs)
        self.encoder = hub_layer

    def call(self, ragged_inputs): # `flatvals` and `rowspl` are baked into the signature in the serialized model
        ret = self.encoder.resolved_object(flatvals=ragged_inputs.flat_values,
                                           rowspl=ragged_inputs.row_splits)
        ret = tf.RaggedTensor.from_row_splits(ret['output_flat'], ret['output_rows'])
        return ret


class RaggedSparseCategoricalCrossEntropy(tf.keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, from_logits=False, reduction='auto'):
        super().__init__(from_logits=from_logits, reduction=reduction)

    def call(self, y_true, y_pred):
        return super().call(y_true.flat_values, y_pred.flat_values)


# @tf.keras.utils.register_keras_serializable()
@keras_register_once(package='Custom', name='RaggedEmbeddingDropout')
class RaggedEmbeddingDropout(tf.keras.layers.Layer):
    """ A Keras layer for embedding dropout which is serializable with ragged tensors """
    def __init__(self, encoder_dp_rate, **kwargs):
        super(RaggedEmbeddingDropout, self).__init__(**kwargs)
        self.trainable = False
        self.encoder_dp_rate = encoder_dp_rate
        self.supports_masking = True
        self.bsize = None
        self._supports_ragged_inputs = True # for compatibility with TF 2.2

    def build(self, input_shape):
        self.bsize = input_shape[0]
        print(">>>> INSIDE BUILD / RaggedEmbDrop <<<<")

    def call(self, inputs, training=None): # inputs is a ragged tensor now
        if training is None:
            training = tf.keras.backend.learning_phase()

        def dropped_embedding():
            """ Drops whole words. Almost, but not 100% the same as dropping them inside the encoder """
            flattened_batch = inputs.flat_values # inputs is a ragged tensor
            row_starts = inputs.row_starts() # size = batch size
            # row_length = input.row.lengths() 
            # bsize = tf.shape(inputs)[0]
            # seq_len = tf.shape(inputs)[1]
            ones = tf.ones((tf.shape(flattened_batch)[0],), dtype=tf.float32)
            dp_mask = tf.nn.dropout(ones, rate=self.encoder_dp_rate)
            dp_mask = tf.cast(tf.cast(dp_mask, tf.bool), tf.float32) # proper zeros and ones
            dropped_flat = tf.multiply(flattened_batch, tf.expand_dims(dp_mask, axis=1)) # axis is 1 because we still haven't restored the number of train examples in a batch
            dropped_out_ragged = tf.RaggedTensor.from_row_starts(dropped_flat, row_starts)
            return dropped_out_ragged

        if training:
            ret = dropped_embedding()
        else:
            ret = array_ops.identity(inputs)
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'encoder_dp_rate': self.encoder_dp_rate})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @tf.keras.utils.register_keras_serializable()
@keras_register_once(package='Custom', name='EmbeddingDropout')
class EmbeddingDropout(tf.keras.layers.Layer):
    """ A Keras layer for embedding dropout """
    def __init__(self, encoder_dp_rate, **kwargs):
        super(EmbeddingDropout, self).__init__(**kwargs)
        self.trainable = False
        self.encoder_dp_rate = encoder_dp_rate
        self.supports_masking = True
        self.bsize = None

    def build(self, input_shape):
        self.bsize = input_shape[0]
        print(">>>> INSIDE BUILD <<<< ")

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        def dropped_embedding():
            """ Drops whole words. Almost, but not 100% the same as dropping them inside the encoder """
            bsize = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            ones = tf.ones((bsize, seq_len), dtype=tf.float32)
            dp_mask = tf.nn.dropout(ones, rate=self.encoder_dp_rate)
            dp_mask = tf.cast(tf.cast(dp_mask, tf.bool), tf.float32) # proper zeros and ones
            dropped = inputs * tf.expand_dims(dp_mask, axis=2)
            return dropped

        if training:
            ret = dropped_embedding()
        else:
            ret = array_ops.identity(inputs)
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'encoder_dp_rate': self.encoder_dp_rate})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @tf.keras.utils.register_keras_serializable()
@keras_register_once(package='Custom', name='TiedDense')
class TiedDense(tf.keras.layers.Layer):
    """ A dense layer with trainable biases and weights fixed (tied) to another dense layer """
    def __init__(self, reference_layer, activation, **kwargs):
        self.ref_layer = reference_layer
        self.biases = None
        self.input_dim = None
        self.output_dim = None
        self.activation_fn = tf.keras.activations.get(activation)
        super().__init__(**kwargs)
        self._supports_masking = self.supports_masking = True

    def build(self, input_shape):
        self.input_dim = self.ref_layer.variables[0].shape[0]
        self.output_dim = self.ref_layer.variables[0].shape[1]
        self.biases = self.add_weight(name='tied_bias',
                                      shape=[self.input_dim],
                                      initializer='zeros')
        super().build(input_shape)

    def call(self, inputs):
        try:
            wx = tf.matmul(inputs, self.ref_layer.variables[0], transpose_b=True)
            z = self.activation_fn(wx + self.biases)
        except:
            tf.print("Warning, warning... - FORWARD PASS GOES TO NULL!")
            # z = tf.matmul(inputs, tf.zeros((self.ref_layer.input_dim, self.ref_layer.output_dim)), transpose_b=True)
            z = tf.matmul(inputs, tf.zeros((self.input_dim, self.output_dim)), transpose_b=True)
        return z

    def compute_output_shape(self, input_shape):
        # tf.print(f"For TIED DENSE the input shape is {input_shape}")
        # return (input_shape[0], tf.shape(self.ref_layer.weights[0])[0])
        return (input_shape[0], self.ref_layer.variables[0].shape[0])
        # return (input_shape[0], 35000)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'reference_layer': self.ref_layer, 'activation': self.activation_fn})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @tf.keras.utils.register_keras_serializable()
@keras_register_once(package='Custom', name='ExplicitMaskGenerator')
class ExplicitMaskGenerator(tf.keras.layers.Layer):
    """ Explicitly return the propagated mask.

        This is useful after serialization where the original _keras_mask object is no longer available.
    """
    def __init__(self, mask_value=None, **kwargs):
        super().__init__(**kwargs)
        self.mask_value = mask_value
        self.supports_masking = False

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        explicit_mask = tf.where(inputs == self.mask_value, False, True)
        explicit_mask = tf.cast(explicit_mask, dtype=tf.bool)
        return explicit_mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'mask_value': self.mask_value})
        return cfg

    @classmethod
    def from_config(cls, config):
        clazz = cls(**config)
        return clazz


# @tf.keras.utils.register_keras_serializable()
@keras_register_once(package='Custom', name='CustomMaskableEmbedding')
class CustomMaskableEmbedding(tf.keras.layers.Embedding):
    """ Enhancement of TF's embedding layer where you can set the custom
        value for the mask token, not just zero. SentencePiece uses 1 for <pad>
        and 0 for <unk> and ULMFiT has adopted this convention too.
    """
    def __init__(self, input_dim, output_dim, embeddings_initializer='uniform',
                 embeddings_regularizer=None, activity_regularizer=None,
                 embeddings_constraint=None, mask_value=None, input_length=None,
                 **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim,
                         embeddings_initializer=embeddings_initializer,
                         embeddings_regularizer=embeddings_regularizer,
                         activity_regularizer=activity_regularizer,
                         embeddings_constraint=embeddings_constraint,
                         input_length=input_length, **kwargs)
        self.mask_value = mask_value
        if self.mask_value is not None:
            self._supports_masking = True
            self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if not self.mask_value:
            return None
        return math_ops.not_equal(inputs, self.mask_value)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'mask_value': self.mask_value})
        return cfg

    @classmethod
    def from_config(cls, config):
        clazz = cls(**config)
        return clazz


# @tf.keras.utils.register_keras_serializable()
@keras_register_once(package='Custom', name='RaggedSpatialDropout1D')
class RaggedSpatialDropout1D(tf.keras.layers.Layer):
    """ A serializable spatial dropout layer that works with RaggedTensors """
    def __init__(self, rate, **kwargs):
        super(RaggedSpatialDropout1D, self).__init__(**kwargs)
        self.trainable = False
        self.rate = rate
        self.supports_masking = True
        self.bsize = None
        self._supports_ragged_inputs = True # for compatibility with TF 2.2

    def build(self, input_shape):
        self.bsize = input_shape[0]
        print(">>>> INSIDE BUILD / RSD<<<< ")

    def call(self, inputs, training=None): # inputs is a ragged tensor now
        if training is None:
            training = tf.keras.backend.learning_phase()

        def dropped_1d():
            """ Spatial 1D dropout which operates on ragged tensors """
            flattened_batch = inputs.flat_values # inputs is a ragged tensor
            row_starts = inputs.row_starts() # size = batch size
            # row_length = input.row.lengths() 
            # bsize = tf.shape(inputs)[0]
            # seq_len = tf.shape(inputs)[1]
            ones = tf.ones((tf.shape(flattened_batch)[1],), dtype=tf.float32)
            dp_mask = tf.nn.dropout(ones, rate=self.rate)
            dp_mask = tf.cast(tf.cast(dp_mask, tf.bool), tf.float32) # proper zeros and ones
            dropped_flat = tf.multiply(flattened_batch, tf.expand_dims(dp_mask, axis=0)) # axis is 0 this time
            dropped_out_ragged = tf.RaggedTensor.from_row_starts(dropped_flat, row_starts)
            return dropped_out_ragged

        # ret = tf.cond(tf.convert_to_tensor(training),
        #               dropped_1d,
        #               lambda: array_ops.identity(inputs))
        if training:
            ret = dropped_1d()
        else:
            ret = array_ops.identity(inputs)
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'rate': self.rate})
        return cfg

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# @tf.keras.utils.register_keras_serializable()
@keras_register_once(package='Custom', name='ConcatPooler')
class ConcatPooler(tf.keras.layers.Layer):
    """ Concatenates the encoder's last hidden state with vectors obtained from MaxPool and AvgPool across timesteps.

        This is idiomatic to ULMFiT for document classification. See the paper for more details.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False
        self._supports_ragged_inputs = False # for compatibility with TF 2.2
    
    def build(self, input_shape):
        print(">>>> INSIDE BUILD / ConcatPooler <<<< ")
    
    def call(self, inputs, training=None, mask=None): # inputs is a fixed-length tensor
        # We cannot use the line below. An tf.keras.layers.LSTMCell wrapped around tf.keras.layers.RNN propagates the last hidden
        # state to the end of the sequence when it encounters padding, whereas a tf.keras.layers.LSTM will insert zeroes.
        # This is quite a gotcha, so we have to compute the position of the last hidden state from Keras mask instead.
        #last_hidden_states = inputs[:, -1, :]

        surely_masked_inputs = tf.where(tf.expand_dims(mask, axis=-1), inputs, tf.zeros_like(inputs)) # always zero on padding
        surely_maxable_inputs = tf.where(tf.expand_dims(mask, axis=-1), inputs, tf.fill(tf.shape(inputs), -np.inf)) # always minus infinity on padding
        num_indices = tf.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=1) # number of tokens
        last_indices = num_indices - 1 # last token index
        batch_index = tf.range(tf.shape(inputs)[0])
        last_hidden_states = tf.gather_nd(inputs, tf.stack([batch_index, last_indices], axis=1))

        #mean_pooled = tf.math.reduce_mean(inputs, axis=1)
        summed = tf.reduce_sum(surely_masked_inputs, axis=1)
        mean_pooled = summed / tf.expand_dims(tf.cast(num_indices, dtype=tf.float32), axis=-1)

        #max_pooled = tf.math.reduce_max(inputs, axis=1)
        max_pooled = tf.math.reduce_max(surely_maxable_inputs, axis=1)

        ret = tf.concat([last_hidden_states, max_pooled, mean_pooled], axis=1)
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2]*3)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# @tf.keras.utils.register_keras_serializable()
@keras_register_once(package='Custom', name='RaggedConcatPooler')
class RaggedConcatPooler(tf.keras.layers.Layer):
    """ Same as ConcatPooler but works with RaggedTensors """
    def __init__(self, inputs_are_flattened=False, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False
        self._supports_ragged_inputs = True # for compatibility with TF 2.2
        self.inputs_are_flattened = inputs_are_flattened # set this to True if using the TFHub version

    def build(self, input_shape):
        print(">>>> INSIDE BUILD / RaggedConcatPooler <<<< ")

    def call(self, inputs, training=None): # inputs is a ragged tensor
        if self.inputs_are_flattened:
            flat_vals = inputs[0]
            row_limits = inputs[1][1:] - 1 # this is row splits from first index minus 1
            last_hidden_states = tf.gather(flat_vals, row_limits)
            ragged_tensor = tf.RaggedTensor.from_row_splits(inputs[0], inputs[1])
        else:
            flat_vals = inputs.flat_values
            row_limits = inputs.row_limits() - 1
            ragged_tensor = inputs
            last_hidden_states = tf.gather(flat_vals, row_limits)

        max_pooled = tf.math.reduce_max(ragged_tensor, axis=1)
        mean_pooled = tf.math.reduce_mean(ragged_tensor, axis=1)
        ret = tf.concat([last_hidden_states, max_pooled, mean_pooled], axis=1)
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2]*3)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def apply_awd_eagerly(encoder_num, awd_rate):
    """ Apply AWD in TF eager mode

        Note: there is also a variant of this function that is serialized into a SavedModel.
        See ExportableULMFiT object for details.
    """
    # tf.print("Applying AWD eagerly")
    rnn1_w = encoder_num.get_layer("AWD_RNN1").variables
    rnn2_w = encoder_num.get_layer("AWD_RNN2").variables
    rnn3_w = encoder_num.get_layer("AWD_RNN3").variables

    w1_mask = tf.nn.dropout(tf.fill(rnn1_w[1].shape, 1-awd_rate), rate=awd_rate)
    rnn1_w[1].assign(w1_mask * rnn1_w[1])

    w2_mask = tf.nn.dropout(tf.fill(rnn2_w[1].shape, 1-awd_rate), rate=awd_rate)
    rnn2_w[1].assign(w2_mask * rnn2_w[2])

    w3_mask = tf.nn.dropout(tf.fill(rnn3_w[1].shape, 1-awd_rate), rate=awd_rate)
    rnn3_w[1].assign(w3_mask * rnn3_w[2])


class AWDCallback(tf.keras.callbacks.Callback):
    """
    Keras-compatible callback which applies AWD after each batch.
    Works with both weights and SavedModel formats
    """
    def __init__(self, *, model_object=None, hub_object=None, awd_rate=0.5):
        super().__init__()
        if not any([model_object, hub_object]) or all([model_object, hub_object]):
            raise ValueError("Pass either `model_object` (for eager mode) or `hub_object`"
                             "(for graph mode), not none, not both.")
        self.model_object = model_object
        self.hub_object = hub_object
        self.awd_rate = awd_rate

    def on_train_batch_begin(self, batch, logs=None):
        if self.hub_object is not None:
            self.hub_object.apply_awd(self.awd_rate)
        else:
            apply_awd_eagerly(self.model_object, self.awd_rate)


# @tf.keras.utils.register_keras_serializable()
@keras_register_once(package='Custom', name='STLRSchedule')
class STLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implementation of slanted triangular learning rates as a Keras LR scheduler.
    You can pass an instance of this class to the optimizer instead of a fixed LR value.
    See ULMFiT paper for the meanings of `cut`, `cut_frac` and `ratio`.
    """
    def __init__(self, lr_max, num_steps, cut_frac=0.1, ratio=32):
        self.lr_max = lr_max   # 0.01
        self.T = num_steps     # 900, which is 90 steps over 10 epochs
        self.cut_frac = cut_frac  # 0.1
        self.cut = math.floor(num_steps * cut_frac)  # 90
        self.ratio = ratio

    def __call__(self, step):
        def warmup(): return step / self.cut
        def cooldown(): return 1 - ((step - self.cut)/(self.cut*(1/(self.cut_frac)-1)))

        p = tf.cond(tf.less(step, self.cut), warmup, cooldown)
        current_lr = self.lr_max * ( (1 + (p*(self.ratio - 1))) / self.ratio)
        return current_lr

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'lr_max': self.lr_max,
                    'num_steps': self.T,
                    'cut_frac': self.cut_frac,
                    'ratio': self.ratio})
        return cfg

    @classmethod
    def from_config(cls, config):
        clazz = cls(**config)
        return clazz


class PredictionProgressCallback(tf.keras.callbacks.Callback):
    """
    Shows a progress bar when calling model.predict()
    """
    def __init__(self, num_steps):
        self.num_steps = num_steps
        self.progress_bar = trange(num_steps)

    def on_predict_batch_begin(self, batch, logs=None):
        self.progress_bar.update()


# The three classes below for fine-tuning the learning rate (CosineAnnealer, OneCycleSheduler and LRFinder)
# are implementations from a notebook by Andrich van Wyk (released under Apache 2.0 license).
# Source: https://www.kaggle.com/avanwyk/tf2-super-convergence-with-the-1cycle-policy
# They are not used in examples, but are left here for experiments with alternatives to triangular learning
# rates.

class CosineAnnealer:

    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos


class OneCycleScheduler(tf.keras.callbacks.Callback):
    """`Callback` that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
    If the model supports a momentum parameter, it will also be adapted by the schedule.
    The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
    only two phases are used and the adaptation is done using cosine annealing.
    In phase 1 the LR increases from `lr_max / div_factor` to `lr_max` and momentum decreases from `mom_max` to `mom_min`.
    In the second phase the LR decreases from `lr_max` to `lr_max / (div_factor * 1e4)` and momemtum from `mom_max` to `mom_min`.
    By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter `phase_1_pct`.
    """

    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps

        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0

        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)], 
                 [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]

        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        try:
            _ = self.model.optimizer._get_hyper('learning_rate')
        except KeyError:
            print(f"WARNING: The optimizer doesn't have the `learning_rate` parameter! Something looks broken...")
        try:
            _ = self.model.optimizer._get_hyper('momentum')
        except KeyError:
            print(f"OneCycleSchduler warning: Your optimizer doesn't have the `momentum` parameter! Try SGD or AdamW instead.")
        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1

        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())

    def get_lr(self):
        try:
            # return tf.keras.backend.get_value(self.model.optimizer.lr)
            # print(f"LR is {self.model.optimizer._get_hyper('learning_rate')}")
            return self.model.optimizer._get_hyper('learning_rate')
        except NameError:
            print("Oops... cannot get a LR! This looks very wrong!")

    def get_momentum(self):
        try:
            return self.model.optimizer._get_hyper('momentum')
            # return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except NameError:
            pass # Not all optimizers have the 'momentum' parameter

    def set_lr(self, lr):
        try:
            self.model.optimizer._set_hyper('learning_rate', lr)
            # tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except KeyError:
            print("Oops... cannot set a LR! This looks broken!")

    def set_momentum(self, mom):
        try:
            self.model.optimizer._set_hyper('momentum', mom)
            # tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except KeyError:
            pass # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]

    def mom_schedule(self):
        return self.phases[self.phase][1]

    def plot(self):
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title('Learning Rate')
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title('Momentum')
        plt.show()


class LRFinder(tf.keras.callbacks.Callback):
    """`Callback` that exponentially adjusts the learning rate after each training batch between `start_lr` and
    `end_lr` for a maximum number of batches: `max_step`. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the `plot` method.
    """

    def __init__(self, start_lr: float = 1e-5, end_lr: float = 10, max_steps: int = 100, smoothing=0.1):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        self.model.optimizer._set_hyper('learning_rate', self.lr)
        # tf.keras.backend.set_value(.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses)
        plt.show()
