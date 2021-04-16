import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import LSTMCell
from awdlstm_tf2 import WeightDropLSTMCell

VOCAB_SIZE=35000
MAX_SEQ_LEN=70

#
#m = tf.keras.models.load_model('ull', custom_objects={'CustomMaskableEmbedding': CustomMaskableEmbedding, 'EmbeddingDropout': EmbeddingDropout, 'WeightDropLSTMCell': WeightDropLSTMCell, 'TiedDense': TiedDense})
#

# au.save('nietrasowany', save_traces=False)


def almost_ulmfit_model(fixed_seq_len=None):

    # layer initializers as per the AWD-LSTM paper
    uniform_initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
    # AWD LSTM cells as per the said paper
    #AWD_LSTM_Cell1 = WeightDropLSTMCell(1150, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    #AWD_LSTM_Cell2 = WeightDropLSTMCell(1150, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    #AWD_LSTM_Cell3 = WeightDropLSTMCell(400, kernel_initializer='glorot_uniform', weight_dropout=0.5)

    AWD_LSTM_Cell1 = LSTMCell(1152, kernel_initializer='glorot_uniform')
    AWD_LSTM_Cell2 = LSTMCell(1152, kernel_initializer='glorot_uniform')
    AWD_LSTM_Cell3 = LSTMCell(400, kernel_initializer='glorot_uniform')
    if fixed_seq_len is None:
        il = tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True)
    else:
        il = tf.keras.layers.Input(shape=(fixed_seq_len,), dtype=tf.int32)
    embedz = CustomMaskableEmbedding(VOCAB_SIZE,
                                     400,
                                     embeddings_initializer=uniform_initializer,
                                     mask_zero=False,
                                     mask_value=1,
                                     name="ulmfit_embeds")
    if fixed_seq_len is None:
        encoder_dropout = RaggedEmbeddingDropout(encoder_dp_rate=0.4, name="ragged_emb_dropout")
        SpatialDrop1DLayer = RaggedSpatialDropout1D
        layer_name_prefix="ragged_"
    else:
        encoder_dropout = EmbeddingDropout(encoder_dp_rate=0.4, name="emb_dropout")
        SpatialDrop1DLayer = tf.keras.layers.SpatialDropout1D
        layer_name_prefix=""
    input_dropout = SpatialDrop1DLayer(0.4, name=f"{layer_name_prefix}inp_dropout")

    rnn1 = tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1")
    rnn1_drop = SpatialDrop1DLayer(0.3, name=f"{layer_name_prefix}rnn_drop1") # yeah, this is quirky, but that's what ULMFit authors propose

    rnn2 = tf.keras.layers.RNN(AWD_LSTM_Cell2, return_sequences=True, name="AWD_RNN2")
    rnn2_drop = SpatialDrop1DLayer(0.3, name=f"{layer_name_prefix}rnn_drop2")

    rnn3 = tf.keras.layers.RNN(AWD_LSTM_Cell3, return_sequences=True, name="AWD_RNN3")
    rnn3_drop = SpatialDrop1DLayer(0.4, name=f"{layer_name_prefix}rnn_drop3")


    m = tf.keras.models.Sequential()
    m.add(il)
    m.add(embedz)
    m.add(encoder_dropout)
    m.add(input_dropout)

    m.add(rnn1); m.add(rnn1_drop)
    m.add(rnn2); m.add(rnn2_drop)
    m.add(rnn3); m.add(rnn3_drop)
    
    #fc_head = tf.keras.layers.TimeDistributed(TiedDense(m.layers[0]))
    #fc_head_dp = tf.keras.layers.Dropout(0.05)
    #fc_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(VOCAB_SIZE, activation='softmax'))
    fc_head = tf.keras.layers.TimeDistributed(TiedDense(reference_layer=embedz, activation='softmax'), name='lm_head_tied')
    fc_head_dp = tf.keras.layers.Dropout(0.05)

    m.add(fc_head)
    m.add(fc_head_dp)
    return m

    # TODO:
    # 1) [DONE] Weight tying
    # 1) [DONE] use 1, not 0 for mask/pad
    # 2) One-cycle policy,
    # 3) Optionally the tokenizer as module,
    # 4) Heads for finetuning LM + Head for text classification
    # 5) Cross-batch statefulness

    #lm_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(VOCAB_SIZE, activation='linear', name="LM_Head"))(rnn3_drop)
    #lm_head_drop = tf.keras.layers.Dropout(0.2)(lm_head)
    #model = tf.keras.models.Model(inputs=il, outputs=lm_head_drop, name="ULMFit Pretraining/Finetuning: TF 2.0 implementation")
    return m


@tf.keras.utils.register_keras_serializable()
class RaggedEmbeddingDropout(tf.keras.layers.Layer):
    def __init__(self, encoder_dp_rate, **kwargs):
        super(RaggedEmbeddingDropout, self).__init__(**kwargs)
        self.trainable = False
        self.encoder_dp_rate = encoder_dp_rate
        self.supports_masking = True
        self.bsize = None
        self._supports_ragged_inputs = True # for compatibility with TF 2.2

    def build(self, input_shape):
        self.bsize = input_shape[0]
        print(">>>> INSIDE BUILD <<<< ")

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
            tf.print(f"{tf.shape(flattened_batch)}")
            ones = tf.ones((tf.shape(flattened_batch)[0],), dtype=tf.float32)
            dp_mask = tf.nn.dropout(ones, rate=self.encoder_dp_rate)
            dp_mask = tf.cast(tf.cast(dp_mask, tf.bool), tf.float32) # proper zeros and ones
            dropped_flat = tf.multiply(flattened_batch, tf.expand_dims(dp_mask, axis=1)) # axis is 1 because we still haven't restored the number of train examples in a batch
            dropped_out_ragged = tf.RaggedTensor.from_row_starts(dropped_flat, row_starts)
            return dropped_out_ragged

        ret = tf.cond(tf.convert_to_tensor(training),
                      dropped_embedding,
                      lambda: array_ops.identity(inputs))
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

@tf.keras.utils.register_keras_serializable()
class EmbeddingDropout(tf.keras.layers.Layer):
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

        ret = tf.cond(tf.convert_to_tensor(training),
                      dropped_embedding,
                      lambda: array_ops.identity(inputs))
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

@tf.keras.utils.register_keras_serializable()
class TiedDense(tf.keras.layers.Layer):
    def __init__(self, reference_layer, activation, **kwargs):
        self.ref_layer = reference_layer
        self.biases = None
        self.activation_fn = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):
        #self.biases = self.add_weight(name='tied_bias',
        #                              shape=[self.ref_layer.weights[0].shape[0]],
        #                              initializer='zeros')
        self.biases = self.add_weight(name='tied_bias',
                                      shape=[self.ref_layer.input_dim],
                                      initializer='zeros')
        super().build(input_shape)

    def call(self, inputs):
        try:
            wx = tf.matmul(inputs, self.ref_layer.weights[0], transpose_b=True)
            z = self.activation_fn(wx + self.biases)
        except:
            print("Warning, warning...")
            z = tf.matmul(inputs, tf.zeros((self.ref_layer.input_dim, self.ref_layer.output_dim)), transpose_b=True)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ref_layer.weights[0].shape[0])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'reference_layer': self.ref_layer, 'activation': self.activation_fn})
        return cfg

    @classmethod
    def from_config(cls, config):
      return cls(**config)

@tf.keras.utils.register_keras_serializable()
class CustomMaskableEmbedding(tf.keras.layers.Embedding):
    """ Enhancement of TF's embedding layer where you can set the custom
        value for the mask token, not just zero. SentencePiece uses 1 for <pad>
        and 0 for <unk> and ULMFiT has adopted this convention too.
    """
    def __init__(self,
            input_dim,
            output_dim,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_value=None,
            input_length=None,
            **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim,
                         embeddings_initializer=embeddings_initializer,
                         embeddings_regularizer=embeddings_regularizer,
                         activity_regularizer=activity_regularizer,
                         embeddings_constraint=embeddings_constraint,
                         input_length=input_length, **kwargs)
        self.mask_value=mask_value

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
      return cls(**config)

@tf.keras.utils.register_keras_serializable()
class RaggedSpatialDropout1D(tf.keras.layers.Layer):
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
            tf.print(f"{tf.shape(flattened_batch)}")
            ones = tf.ones((tf.shape(flattened_batch)[1],), dtype=tf.float32)
            dp_mask = tf.nn.dropout(ones, rate=self.rate)
            dp_mask = tf.cast(tf.cast(dp_mask, tf.bool), tf.float32) # proper zeros and ones
            dropped_flat = tf.multiply(flattened_batch, tf.expand_dims(dp_mask, axis=0)) # axis is 0 this time
            dropped_out_ragged = tf.RaggedTensor.from_row_starts(dropped_flat, row_starts)
            return dropped_out_ragged

        ret = tf.cond(tf.convert_to_tensor(training),
                      dropped_1d,
                      lambda: array_ops.identity(inputs))
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

