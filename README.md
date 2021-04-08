# Implementation of AWD-LSTM for Tensorflow 2.0

This repo contains a Keras-compatible AWD-LSTM cell in its most basic form. This means:

* no fancy variational dropout
* no peephole connections

All there is is just the implementation of dropout on recurrent (U) matrices. This version seems to have been used in ULMFit, so it's a good point from which to start porting ULMFit to TF as well.

## Usage
Notice the additional **weight_dropout** parameter in the cell's constructor.

```
import tensorflow as tf
from awdlstm_tf2 import WeightDropLSTMCell

VOCAB_SIZE=5000
EMB_DIM=400

AWD_cell1 = WeightDropLSTMCell(1150, kernel_initializer='glorot_uniform', weight_dropout=0.5)

input_layer = tf.keras.layers.Input(shape=(70,))
embeddings = tf.keras.layers.Embedding(VOCAB_SIZE, EMB_DIM)(input_layer)
recurrent_layer = tf.keras.layers.RNN(AWD_cell1, return_sequences=True)(embeddings)
# .... proceed with building a Keras model as usual .....

```

## ULMFit language model for TensorFlow 2.0
There are many more regularization and preprocessing quirks needed to port ULMFit than just AWD-LSTM. For inference, you can even just export weights from Fast.ai's implementation and read them into Keras LSTM layers - there is no dropout during inference, so your output will be numerically compatible. However, for training you would need to replicate all the quirks plus use the same optimizer as the paper's authors.

I tried to be as faithful as I possibly could to both the paper and the Fast.ai's implementation in PyTorch. Here are all the regularization tricks I have implemented for the **language modelling / fine-tuning**:

* embedding dropout
* input dropout
* weight dropout (AWD-LSTM)
* weight tying between embeddings and the LM head

See [almost_ulmfit_tf2.py](almost_ulmfit_tf2.py) for a version that includes the language modelling head. I haven't really tested yet what happens when you run it on an actual LM task. This is because ULMFit needs a carefully chosen optimizer (slanted triangular learning rates, later replaced with something called one-cycle policy). I'm not really sure what will happen if you run it with out-of-the-box Adam optimizer from Keras. The model will most likely overfit, but go ahead and try it yourself if you have GPU time.

## TODO: ULMFiT for text classification for TensorFlow 2.0
Not yet done. This is quite straightforward in terms of tensor operations, but difficult when it comes to optimizers. I'll get to it 'soon'.

## TODO: ULMFiT for sequence tagging for TensorFlow 2.0
Frankly speaking, I haven't seen anyone (not even ULMFiT's authors) using this model for things like NER. Maybe the model which is excellent on text classification is so bad at sequence tagging that the authors didn't even want to mention the results? I'll try to investigate it 'a little later'.