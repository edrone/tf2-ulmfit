import tensorflow as tf
from tensorflow.python.ops import array_ops
from awdlstm_tf2 import WeightDropLSTMCell

VOCAB_SIZE=10

def almost_ulmfit_model():

    # layer initializers as per the AWD-LSTM paper
    uniform_initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
    # AWD LSTM cells as per the said paper
    AWD_LSTM_Cell1 = WeightDropLSTMCell(1150, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    AWD_LSTM_Cell2 = WeightDropLSTMCell(1150, kernel_initializer='glorot_uniform', weight_dropout=0.5)
    AWD_LSTM_Cell3 = WeightDropLSTMCell(400, kernel_initializer='glorot_uniform', weight_dropout=0.5)

    il = tf.keras.layers.Input(shape=(70,))
    embedz = tf.keras.layers.Embedding(VOCAB_SIZE,
                                       400,
                                       embeddings_initializer=uniform_initializer,
                                       mask_zero=True)(il)
    # encoder_dropout = EmbeddingDropout(0.4)(embedz)
    #input_dropout = tf.keras.layers.SpatialDropout1D(0.4)(encoder_dropout)
    input_dropout = tf.keras.layers.SpatialDropout1D(0.4)(embedz)
                                                  
    rnn1 = tf.keras.layers.RNN(AWD_LSTM_Cell1, return_sequences=True, name="AWD_RNN1")(input_dropout)
    rnn1_drop = tf.keras.layers.SpatialDropout1D(0.3)(rnn1) # yeah, this is quirky, but that's what ULMFit authors propose
    
    rnn2 = tf.keras.layers.RNN(AWD_LSTM_Cell2, return_sequences=True, name="AWD_RNN2")(rnn1_drop)
    rnn2_drop = tf.keras.layers.SpatialDropout1D(0.3)(rnn2)

    rnn3 = tf.keras.layers.RNN(AWD_LSTM_Cell3, return_sequences=True, name="AWD_RNN3")(rnn2_drop)
    rnn3_drop = tf.keras.layers.SpatialDropout1D(0.4)(rnn3)

    # TODO: WEIGHT TYING

    lm_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(VOCAB_SIZE, activation='linear', name="LM_Head"))(rnn3_drop)
    lm_head_drop = tf.keras.layers.Dropout(0.2)(lm_head)
    model = tf.keras.models.Model(inputs=il, outputs=lm_head_drop, name="ULMFit Pretraining/Finetuning: TF 2.0 implementation")
    return model


# class EmbeddingDropout(tf.keras.layers.Layer):
#     """ THIS LAYER IS BROKEN - DON'T USE IT! I'LL FIX IT 'SOON' """
#     def __init__(self, rate):
#         super(EmbeddingDropout, self).__init__()
#         self.trainable = False
#         self.rate = rate
#         self.supports_masking = True
#         self.bsize = None
# 
#     def build(self, input_shape):
#         self.bsize = input_shape[0]
#         print(">>>> INSIDE BUILD <<<< ")
# 
#     def call(self, inputs, training=None):
#         if training is None:
#             training = tf.keras.backend.learning_phase()
#         print(">>>> INSIDE CALL <<<< ")
# 
#         def dropped_embedding():
#             num_words = inputs.shape[-1]
#             print(f">>>>> INPUTS: {inputs} {num_words} <<<<<<<")
#             num_embeds_to_drop = int(self.rate * num_words)
#             drop_indices = tf.random.uniform((num_embeds_to_drop,), maxval=num_words, dtype=tf.int32)
#             drop_mask_1d = tf.map_fn(lambda p: 1 if p not in drop_indices else 0, 
#                                      tf.range((num_words,1), dtype=tf.int32))
#             drop_mask_1d = tf.reshape(tf.cast(drop_mask_1d, dtype=tf.float32), (num_words, 1))
#             row_wise_dropped = tf.multiply(inputs, drop_mask_1d)
#             return row_wise_dropped
# 
#         ret = tf.cond(training,
#                       dropped_embedding,
#                       lambda: array_ops.identity(inputs))
#         return ret
# 
#     def compute_output_shape(self, input_shape):
#         return input_shape

