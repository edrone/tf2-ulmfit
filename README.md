# Implementation of AWD-LSTM and ULMFiT for Tensorflow 2.0

## The AWD-LSTM cell
This repo contains a Keras-compatible AWD-LSTM cell used in ULMFiT in its most basic form. This means:

* no fancy variational dropout
* no peephole connections

All there is is just the implementation of weight dropout on recurrent (U) matrices. At the moment we refresh the weight dropout mask every batch (not every timestep as the default implementation does) - this saves a considerable amount of memory and looks compatible with what FastAI did in their implementation.

### Usage
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
* weight tying between embeddings and the LM head (for the language modelling / fine-tuning task)

See `tf2_ulmfit_encoder` function in [ulmfit_tf2.py](ulmfit_tf2.py). This function is called as follows: 
```
lm_model_num, encoder_num, outmask_num, spm_encoder_model = tf2_ulmfit_encoder(fixed_seq_len, spm_model_file`)
```

and as you can see it returns **four** Keras models:

1. `lm_model_num` - an ULMFiT encoder with a language modelling head (embeddings + 3 RNN layers + LM head with tied weights) 
2. `encoder_num` - an ULMFiT encoder only (embeddings + 3 RNN layers)
3. `outmask_num` - explicit array which shows how a fixed-sequence model was padded and masked.
4. `spm_encoder_model` - a tensorflow_text compatible version of a Sentencepiece tokenizer/numericalizer. 

Additionally, ULMFit needs a carefully chosen optimizer (slanted triangular learning rates, later replaced with something called one-cycle policy). This is not currently implemented, but running the model with an out-of-the-box Adam optimizer from Keras appears to work too (well, it probably overfits).

## TODO: ULMFiT for text classification for TensorFlow 2.0
Not yet done. This is quite straightforward in terms of tensor operations, but difficult when it comes to optimizers. I'll get to it 'soon'.

## ULMFiT for sequence tagging for TensorFlow 2.0
Frankly speaking, I haven't seen anyone (not even ULMFiT's authors) using this model for things like Named Entity Recognition (NER) or Part-of-Speech Tagging (POS). From what I know this is the first implementation of ULMFiT for Named Entity Recognition / sequence tagging. And it seems to work really well. Example usage:

```
CP_PATH=path_to_checkpoint_exported_from_FastAI
tagger, spm_proc = ulmfit_sequence_tagger(num_classes=3, \
                   pretrained_weights=CP_PATH, \
                   fixed_seq_len=768, \
                   spm_model_file='pl_wiki100-sp35k.model', \
                   also_return_spm_encoder=True)
```
You can now run `tagger.summary()`, `tagger.fit(x, y, epochs=1, callbacks=[])` like you do with any keras model.

## Serializing ULMFiT to a SavedModel and Tensorflow Hub format