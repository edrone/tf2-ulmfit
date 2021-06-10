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

## Serializing to a SavedModel format
WIP

<hr   style="border:2px solid blue"/>

# Using serialized ULMFiT model from Tensorflow Hub

## Deserializing from a SavedModel and Tensorflow Hub format

```
import tensorflow_text as text
import tensorflow_hub as hub

ulmfit_bundle = hub.load('path_to_saved_model')
example_text = tf.constant(["All men are born free and equal. [SEP] "\
                            "They are endowed with reason and conscience " \
                            "and should act towards one another in a spirit "\
                            "of brotherhood."])
```

The deserialized SavedModel / TF Hub object serves as a wrapper for the proper model, which can be accessed using three **signatures**:

* **string_encoder** - accepts strings, returns hidden states.
   Subword tokenization, conversion of tokens to indices, adding BOS and EOS markers and padding is done automatically.
* **numericalized_encoder** - accepts numericalized input, returns hidden states.
   You can use this signature if you already have the numerical representation of your data.
* **spm_processor** - accepts strings, performs tokenization, conversion of tokens to indices, add BOS and EOS markers and padding.
   Use this signature if you only want to obtain token IDs.



#### The string_encoder signature
This signature returns a dictionary with three keys: `output` contains the hidden states for subsequent tokens, `numericalized` contains subword token IDs and `mask` shows padding (True = token, False = padding).
```
ulmfit_encoder = hub.KerasLayer(ulmfit_bundle.signatures['string_encoder'], trainable=True)
ret = ulmfit_encoder(example_text)
print(ret)
{'output': <tf.Tensor: shape=(1, 512, 400), dtype=float32, numpy=
array([[[ 2.1791911e-04, -3.6128119e-04,  1.6183863e-04, ...,         
         -2.8810455e-04, -1.2454562e-04, -2.1535352e-04],             
        [ 4.5642123e-04, -5.8338832e-04,  3.6983314e-04, ...,         
         -6.2256883e-04, -2.2939572e-04, -4.9976230e-04],             
        [ 5.7463942e-04, -6.3404167e-04,  7.6492300e-04, ...,         
         -1.1259192e-03, -3.5521804e-05, -1.0689020e-03],             
        ...,                                                          
        [ 1.4855270e-03,  1.7452225e-03,  2.3215336e-03, ...,         
          7.3179911e-04,  3.2494958e-03,  1.5500648e-04],             
        [ 1.4855270e-03,  1.7452225e-03,  2.3215336e-03, ...,         
          7.3179911e-04,  3.2494958e-03,  1.5500648e-04],             
        [ 1.4855270e-03,  1.7452225e-03,  2.3215336e-03, ...,         
          7.3179911e-04,  3.2494958e-03,  1.5500648e-04]]], dtype=float32)>, 
'numericalized': <tf.Tensor: shape=(1, 512), dtype=int32, numpy=
array([[    2, 11274,  9769,  2402,    47, 22706, 12507, 34930,  3122,
          248,     0, 34941,   168, 34948,     3,     2,  1988, 34938,
         2402,  4218,  7574, 34940, 18580,   178,   146,    61,  3122,
          253,  2200, 12626,  3122,    16,  1160, 25882,    76,   454,
         6311,  1200, 34935,  2470, 13251, 16065,   235,    76,  4913,
        16254,   942,    47,   782,  3975,  1160,   538, 34948,     3,
            1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1, ...]],
      dtype=int32)>,
'mask': <tf.Tensor: shape=(1, 512), dtype=bool, numpy=
array([[ True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        ...]]>}
		
```
#### The numericalized_encoder signature
This signature runs the same model as string_encoder and returns only the `output` and `mask` keys.

```
fake_input = tf.constant([[2, 10, 20, 30, 3] + [1]*507]) # one sentence padded to 512 token IDs.
ulmfit_encoder_numericalized = hub.KerasLayer(ulmfit_bundle.signatures['numericalized_encoder'], trainable=True)
ret = ulmfit_encoder_numericalized(fake_input)
```


#### The spm_processor signature
This signature returns the `numericalized` and `mask` keys.

```
ulmfit_spm_processor = hub.KerasLayer(ulmfit_bundle.signatures['spm_processor'], trainable=False)
ret = ulmfit_spm_processor(example_text)
print(ret)
```