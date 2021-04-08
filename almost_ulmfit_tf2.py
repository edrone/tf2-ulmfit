import tensorflow as tf
from tensorflow.python.ops import array_ops
from awdlstm_tf2 import WeightDropLSTMCell

VOCAB_SIZE=35000
MAX_SEQ_LEN=70

def almost_ulmfit_model():

    # layer initializers as per the AWD-LSTM paper
    uniform_initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
    # AWD LSTM cells as per the said paper
    AWD_LSTM_Cell1 = WeightDropLSTMCell(1150, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    AWD_LSTM_Cell2 = WeightDropLSTMCell(1150, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    AWD_LSTM_Cell3 = WeightDropLSTMCell(400, kernel_initializer='glorot_uniform', weight_dropout=0.5)

    il = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,))
    embedz = tf.keras.layers.Embedding(VOCAB_SIZE,
                                       400,
                                       embeddings_initializer=uniform_initializer,
                                       mask_zero=True,
                                       name="ulmfit_embeds")
    encoder_dropout = EmbeddingDropout(encoder_dp_rate=0.4, name="emb_dropout")
    input_dropout = tf.keras.layers.SpatialDropout1D(0.4, name="inp_dropout")

    rnn1 = tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1")
    rnn1_drop = tf.keras.layers.SpatialDropout1D(0.3, name="rnn_drop1") # yeah, this is quirky, but that's what ULMFit authors propose

    rnn2 = tf.keras.layers.RNN(AWD_LSTM_Cell2, return_sequences=True, name="AWD_RNN2")
    rnn2_drop = tf.keras.layers.SpatialDropout1D(0.3, name="rnn_drop2")

    rnn3 = tf.keras.layers.RNN(AWD_LSTM_Cell3, return_sequences=True, name="AWD_RNN3")
    rnn3_drop = tf.keras.layers.SpatialDropout1D(0.4, name="rnn_drop3")


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
    fc_head = TiedDense(reference_layer=embedz, activation='softmax', name='lm_head_tied')
    fc_head_dp = tf.keras.layers.Dropout(0.05)

    m.add(fc_head)
    m.add(fc_head_dp)

    # TODO:
    # 1) [DONE] Weight tying
    # 2) One-cycle policy,
    # 3) Optionally the tokenizer as module,
    # 4) Heads for finetuning LM + Head for text classification
    # 5) Cross-batch statefulness

    #lm_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(VOCAB_SIZE, activation='linear', name="LM_Head"))(rnn3_drop)
    #lm_head_drop = tf.keras.layers.Dropout(0.2)(lm_head)
    #model = tf.keras.models.Model(inputs=il, outputs=lm_head_drop, name="ULMFit Pretraining/Finetuning: TF 2.0 implementation")
    return m


class EmbeddingDropout(tf.keras.layers.Layer):
    """ THIS LAYER IS BROKEN - DON'T USE IT! I'LL FIX IT 'SOON' """
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

        ret = tf.cond(training,
                      dropped_embedding,
                      lambda: array_ops.identity(inputs))
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape

class TiedDense(tf.keras.layers.Layer):
    def __init__(self, reference_layer, add_bias=True, activation=None, **kwargs):
        self.ref_layer = reference_layer
        self.biases = None
        self.activation_fn = tf.keras.activations.get(activation)
        self.add_bias = add_bias
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.add_bias is True:
            self.biases = self.add_weight(name='tied_bias',
                                          shape=[self.ref_layer.weights[0].shape[0]],
                                          initializer='zeros')
        super().build(input_shape)

    def call(self, inputs):
        wx = tf.matmul(inputs, self.ref_layer.weights[0], transpose_b=True)
        z = self.activation_fn(wx + self.biases)
        return z
