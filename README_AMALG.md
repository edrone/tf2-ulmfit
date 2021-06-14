# ULMFiT for Tensorflow 2.0

Table of contents

[TOC]



## 1. Introduction (and why we still need FastAI)

This repository contains scripts used to pretrain ULMFiT language models in the FastAI framework, convert the result to a Keras model usable with Tensorflow 2.0, and fine-tune on downstream tasks in Tensorflow.

Please note that whereas you can train any encoder head (document classification, sequence tagging, encoder-decoder etc.) in Tensorflow, the pretraining and fine-tuning of a generic language model should be done only in FastAI. This is because FastAI was written by ULMFiT's authors and contains all the important implementation details that might be omitted in the paper. Porting all these details to another framework is a big challenge. However, once you have the encoder weights trained in a proper way, the setup and hyperparameters of your downstream tasks can diverge from the original implementation (at the expense of regularization).

Basically, ULMFiT is just 3 layers of a unidirectional LSTM network plus many regularization methods. We were successful in porting the following regularization techniques to TF2:

* encoder dropout
* input dropout
* RNN dropout
* weight dropout (AWD) - must be called manually or via a KerasCallback
* slanted triangular learning rates - available as a subcass of  `tf.keras.optimizers.schedules.LearningRateSchedule`

The following techniques are NOT ported:

* batch normalization with the document classification head - we could not get the model to converge on any validation set.
* mysterious calls to undocumented things in FastAI like `rnn_cbs(alpha=2, beta=1)`
* the one-cycle policy for finding the learning rate - this is implemented in newer versions of FastAI; instead we kept the slanted triangular learning rate scheduler as described in the original paper.



## 2. Just give me the pretrained models

Sure. You can download them for English and Polish in three different formats:

* **TF 2.0 SavedModel** - available via Tensorflow Hub as a standalone module. This is great because you don't need any external code (including this repo) to build your own classifiers.
* **Keras weights** - you can build a Keras encoder model using code from this repo and restore the weights via `model.load_weights(...)`. This can be handy if you need to tweak some parameters that were fixed by the paper's authors.
* **FastAI .pth** **state_dict** - the original file which you can convert to a TF 2.0 models with the `convert_fastai2keras.py` script.

All our models were trained on Wikipedia (the datasets were very similar, though not identical, to Wikitext-103) and use Sentencepiece to tokenize input strings into subwords.

Here are the links:

| Model            | TF 2.0 SavedModel | Keras weights | FastAI .pth file | Sentencepiece vocabulary models |
| ---------------- | ----------------- | ------------- | ---------------- | ------------------------------- |
| en-sp35k-cased   |                   |               |                  |                                 |
| en-sp35k-uncased |                   |               |                  |                                 |
| pl-sp50k-cased   |                   |               |                  |                                 |
| pl-sp50k-uncased |                   |               |                  |                                 |



## 3. The encoder

The encoder transforms batches of strings into sequences of vectors representing sentences. Each token is represented as a 400-dimensional vector in the encoder's output.



### 3.1. Tokenization and numericalization

We use [Sentencepiece](https://github.com/google/sentencepiece) to tokenize the input text into subwords. To convert tokens into their IDs (numericalization) you can use the downloaded vocabulary files directly with Python's `sentencepiece` module or its Tensorflow wrapper available in `tensorflow_text` as described in [this manual](https://www.tensorflow.org/tutorials/tensorflow_text/subwords_tokenizer). In line with FastAI's implementation, our vocabularies contain the following special indices:

* 0 - `<unk>`
* 1 - `<pad>` (note that this is unlike Keras where the default padding index is `0`)
* 2 - `<s>` (BOS)
* 3 - `</s>`(EOS)

We also provide a Keras layer object called `SPMNumericalizer` which you can instantiate with a path to the `.spm` file. This is convenient if you just need to process a text dataset into token IDs without worrying about the whole mechanics of vocabulary building:

```
import tensorflow as tf
from ulmfit_tf2 import SPMNumericalizer

spm_processor = SPMNumericalizer(name='spm_layer',
                                 spm_path='enwiki100-cased-sp35k.model',
                                 add_bos=True,
                                 add_eos=True)
print(spm_processor(tf.constant(['Hello, world'], dtype=tf.string)))
<tf.RaggedTensor [[2, 6753, 34942, 34957, 770, 3]]>
```

As you can see, the `SPMNumericalizer` object can even add BOS/EOS markers to each sequence. This can be seen in the output - the numericalized sequence begins with `2` and ends with `3`.



### 3.2. Fixed-length vs variable-length sequences

In the previous section you see that sequences are numericalized into RaggedTensors containing variable length sequences. All the scripts, classes and functions in this repository operate on RaggedTensors by default. This is also the type of input data used by the SavedModel modules available from Tensorflow Hub.

However, if for some reasons you prefer to work with fixed length tensors and padding, you can pass a `fixed_seq_len` parameter. This will truncate and pad all inputs to the specified length:

```
spm_processor = SPMNumericalizer(name='spm_layer',
                                 spm_path='enwiki100-cased-sp35k.model',
                                 add_bos=True,
                                 add_eos=True,
                                 fixed_seq_len=70)
print(spm_processor(tf.constant(['Hello, world'], dtype=tf.string)))
tf.Tensor(
[[    2  6753 34942 34957   770     3     1     1     1     1     1     1
      1     1     1     1     1     1     1     1     1     1     1     1
      1     1     1     1     1     1     1     1     1     1     1     1
      1     1     1     1     1     1     1     1     1     1     1     1
      1     1     1     1     1     1     1     1     1     1     1     1
      1     1     1     1     1     1     1     1     1     1]], shape=(1, 70), dtype=int32)
```

The `fixed_seq_len` parameter is available not only in `SPMNumericalizer`, but also everywhere in this repo where sequence length is relevant. In this guide we prefer the convenience of RaggedTensors and do not use this parameter.



### 3.2. Obtaining the RNN encoder and restoring pretrained weights

You can get an instance of a trainable `tf.keras.Model` containing the encoder by calling the `tf2_ulmfit_encoder` function like this:

```
import tensorflow as tf
from ulmfit_tf2 import tf2_ulmfit_encoder
spm_args = {'spm_model_file': 'enwiki100-cased-sp35k.model',
            'add_bos': True,
            'add_eos': True}
lm_num, encoder_num, mask_num, spm_encoder_model = tf2_ulmfit_encoder(spm_args=spm_args, flatten_ragged_outputs=False)
```

Note that this function returns four objects (all of them instances of `tf.keras.Model`) with **`encoder_num`** being the actual encoder. You can view its structure just like any other Keras model by calling the `summary` method:

```
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
ragged_numericalized_input ( [(None, None)]            0         
_________________________________________________________________
ulmfit_embeds (CustomMaskabl (None, None, 400)         14000000  
_________________________________________________________________
ragged_emb_dropout (RaggedEm (None, None, 400)         0         
_________________________________________________________________
ragged_inp_dropout (RaggedSp (None, None, 400)         0         
_________________________________________________________________
AWD_RNN1 (RNN)               (None, None, 1152)        7156224   
_________________________________________________________________
ragged_rnn_drop1 (RaggedSpat (None, None, 1152)        0         
_________________________________________________________________
AWD_RNN2 (RNN)               (None, None, 1152)        10621440  
_________________________________________________________________
ragged_rnn_drop2 (RaggedSpat (None, None, 1152)        0         
_________________________________________________________________
AWD_RNN3 (RNN)               (None, None, 400)         2484800   
_________________________________________________________________
ragged_rnn_drop3 (RaggedSpat (None, None, 400)         0         
=================================================================
Total params: 34,262,464
Trainable params: 34,262,464
Non-trainable params: 0
_________________________________________________________________

```

Let's now see how we can get the sentence representation. First, we need some texts converted to token IDs. We can use the **`spm_encoder_model`** to obtain it, then all we need to do is pass these IDs to **encoder_num**:

```
text_ids = spm_encoder_model(tf.constant(['Cat sat on a mat', 'And spat'], dtype=tf.string))
vectors = encoder_num(text_ids)
print(vectors.shape)      # (2, None, 400)
print(vectors[0].shape)   # (7, 400)
print(vectors[0].shape)   # (5, 400)
```

As you can see, the encoder outputs a RaggedTensor. There are two sentences in our "batch", so the zeroeth dimension is 2. Sequences are of different lengths (seven and five subwords, including BOS/EOS markers) so the first dimension is `None`. Finally, each output hidden state is represented by 400 floats, so the third dimension is 400.

The other two objects returned by `tf2_ulmfit_encoder` are:

* **`lm_num`** - the encoder with a language modelling head on top. We have followed the ULMFiT paper here and implemented **weight tying** - the LM head's weights (but not biases) are tied to the embedding layer. You will probably only want this for the next-token prediction demo.
* **mask_num** - returns a mask tensor where `True` means a normal token and `False` means padding. This is only relevant in custom models and for Keras mask propagation between layers.

You now have an ULMFiT encoder model with randomly initialized weights. Sometimes this is sufficient for very simple tasks, but generally you will probably want to restore the pretrained weights. This can be done using standard Keras function `load_weights` in the same way as you do with all other Keras models. You just need to provide a path to the directory containing the `checkpoint` and the model name (the `.expect_partial()` bit tells Keras to restore as much as it can from checkpoint and ignore the rest. This quenches some warnings about the optimizer state.):

```encoder_num.load_weights('keras_weights/enwiki100_20epochs_35k_cased').expect_partial()```

It is also possible to restore the encoder from a local copy of a SavedModel directory. This is a little more involved and you will lose the information about all those prettily printed layers, but see the function `ulmfit_rnn_encoder_hub` in [ulmfit_tf2_heads.py](ulmfit_tf2_heads.py) if you are interested in this use case.



## 4. How to use ULMFiT with some typical NLP tasks

### 4.1. Common parameter names used in this repo

WIP

### 4.2. Document classification (the ULMFiT way)

WIP

### 4.3. Document classification (the classical way)

WIP

### 4.4. Regressor

WIP

### 4.5. Sequence tagger

WIP - old text from another file:

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



## 5. Pretraining your own language model from scratch

### 5.1. Data preparation

WIP

### 5.2. Sentencepiece vocabulary

WIP

### 5.3. Training in FastAI

WIP

### 5.4. Exporting to Tensorflow 2.0 and serializing the ExportableULMFiTRagged class

WIP

### 5.5. Perplexity evaluation

WIP



## 6. References and acknowledgements

NCBIR grant.









