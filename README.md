# ULMFiT for Tensorflow 2.0

Table of contents

[TOC]



## 1. Introduction (and why we still need FastAI)

This repository contains scripts used to pretrain ULMFiT language models in the FastAI framework, convert the result to a Keras model usable with Tensorflow 2.0, and fine-tune on downstream tasks in Tensorflow.

Please note that whereas you can train any encoder head (document classification, sequence tagging, encoder-decoder etc.) in Tensorflow, the pretraining and fine-tuning of a generic language model should be done only in FastAI. This is because FastAI was written by ULMFiT's authors and contains all the important implementation details that might be omitted in the paper. Porting all these details to another framework is a big challenge. But having the encoder weights trained in a proper way and available in TF still allows you to take advantage of transfer learning for downstream tasks, even if your hyperparameters are suboptimal.

Basically, ULMFiT is just 3 layers of a unidirectional LSTM network plus many regularization methods. We were successful in porting the following regularization techniques to TF2:

* encoder dropout
* input dropout
* RNN dropout
* weight dropout (AWD) - must be called manually or via a KerasCallback
* slanted triangular learning rates - available as a subclass of  `tf.keras.optimizers.schedules.LearningRateSchedule`

The following techniques are NOT ported:

* the LR finder and one-cycle policy for setting the learning rate schedule - this is implemented in newer versions of FastAI; instead we kept the slanted triangular learning rate scheduler as described in the original paper. There are many existing implementations of both LRFinder and 1-cycle policy for Keras available on the internet, though.
* gradual unfreezing - you can very easily control this yourself by setting the `trainable` attribute on successive Keras layers
* mysterious calls to undocumented things in FastAI like `rnn_cbs(alpha=2, beta=1)`



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



### 3.3. Obtaining the RNN encoder and restoring pretrained weights

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



## 4. How to use ULMFiT for Tensorflow with some typical NLP tasks

In the [examples](examples) directory we are providing training scripts which illustrate how the ULMFiT encoder can be used for a variety of downstream tasks. All the scripts are executable from the command line as Python modules (`python -m examples.ulmfit_tf_text_classifer --help`). After training models on custom data, you can run these scripts with the `--interactive` switch which allows you to type text in the console and display predictions.

### 4.1. Common parameter names used in this repo

The `main` method of each example script accepts a single parameter called `args` which is basically a configuration dictionary created from arguments passed in the command line. Here is a list of the most common arguments you will encounter:

* Data:
  * `--train-tsv / -- test-tsv ` - paths to source files containing training and test/validation data. For classification/regression tasks the input format is a TSV file with a header. For sequence tagging see below.
  * `--data-column-name` - name of the column with input data
  * `--gold-column-name` - name of the column with labels
  * `--label-map` - path to a text (classifier) or json (sequence tagger) file containing labels.
* Tokenization and numericalization:
  * `--spm-model-file` - path to a Sentencepiece .model file
  * `--fixed-seq-len` - if set, input data will be truncated or padded to this number of tokens. If unset, variable-length sequences and RaggedTensors will be used
  * `--max-seq-len` - maximal number of tokens in a sequence. This should generally be set to some sensible value for your data, even if you use RaggedTensors, because one maliciously long sequence can cause an OOM error in the middle of training.
* Restoring model weights and saving the finetuned version:
  * `--model-weights-cp` - path to a local directory where the pretrained encoder weights are saved
  * `--model-type` - what to expect in the directory given in the previous parameter. If set to `from_cp`, the script will expect Keras weights (in this case, provide the checkpoint name as well). If set to `from_hub`, it will expect SavedModel files.
  * `--out-path` - where to save the model's weights after the training completes.
* Training:
  * `--num-epochs` - number of epochs
  * `--batch-size` - batch size
  * `--lr` - peak learning rate for the slanted triangular learning rate scheduler
  * `--awd-off` - disables AWD regularization
  * `--save-best` - if set, the training script will save the model with the best accuracy score on the test/validation set



### 4.2. Document classification (the ULMFiT way - `ulmfit_tf_text_classifier.py`)

This script attempts to replicate the document classifier architecture from the original ULMFiT paper. On top of the encoder there is a layer that concatenates three vectors:

* the max-pooled sentence vector
* the average-pooled sentence vector
* the encoder's last hidden state

This representation is then passed through a 50-dimensional Dense layer. The last layer has a softmax activation and many neurons as there are classes. One issue we encountered here is batch normalization, which is included in the original paper, but which we were not able to use in Tensorflow. When adding BatchNorm to the model we found that we could not get it to converge on any validation set, so it is disabled in our scripts.

Example invocation:

```
python -m examples.ulmfit_tf_text_classifier \
          --train-tsv examples_data/sent200.tsv \
          --data-column-name sentence \
          --gold-column-name sentiment \
          --label-map examples_data/document_classification_labels.txt \
          --model-weights-cp keras_weights/enwiki100_20epochs_35k_uncased \
          --model-type from_cp \
          --spm-model-file enwiki100-uncased-sp35k.model \
          --max-seq-len 300 \
          --num-epochs 12 \
          --batch-size 32 \
          --lr 0.007 \
          --out-path ./sent200trained \
          --save-best
```

Now your classifier is ready in the `sent200trained` directory. The above command trains a classifier on a toy dataset and is almost guaranteed to overfit, but do give it a try with a demo:

```
python -m examples.ulmfit_tf_text_classifier \
          --label-map examples_data/document_classification_labels.txt \
          --model-weights-cp sent200trained/best_checkpoint/best \
          --model-type from_cp \
          --spm-model-file enwiki100-uncased-sp35k.model \
          --max-seq-len 300 \
          --interactive

Paste a document to classify: this is the most depressing film i've ever seen . so boring i left before it finished .
[('POSITIVE', 0.08279895782470703), ('NEGATIVE', 0.917201042175293)]
Classification result: P(NEGATIVE) = 0.8749799728393555
Paste a document to classify: this is the most fascinating film i've ever seen . so captivating i wish it went for another hour .
[('POSITIVE', 0.998953104019165), ('NEGATIVE', 0.0010468183318153024)]
Classification result: P(POSITIVE) = 0.998953104019165
```



### 4.3. Document classification (the classical way `ulmfit_tf_lasthidden_classifier.py`)

This script shows you how you can use the sequence's last hidden state to build a document classifier. We found that its performance was far worse with our pretrained models than the performance of a classifier described in the previous section. We suspect this is because the model was pretrained using a sentence-tokenized corpus with EOS markers at the end of each sequence. To be coherent, we also passed the EOS marker to the classification head in this script, but apparently the recurrent network isn't able to store various sentence "summaries" in an identical token. We nevertheless leave this classification head in the repo in case anyone wanted to investigate potential bugs.

From a technical point of view obtaining the last hidden state is somewhat challenging with RaggedTensors. It turns out we cannot use -1 indexing (`encoder_output[:, -1, :]`) as we would normally do with fixed-length tensors. See the function `ulmfit_last_hidden_state` in [ulmfit_tf2_heads.py](ulmfit_tf2_heads.py) for a workaround.

The invocation is identical as in the previous section.



### 4.4. Regressor (`ulmfit_tf_regressor.py`)

Sometimes instead of predicting a label from a fixed set of classes, you may want to predict a number. For instance, instead of classifying a hotel review as 'positive', 'negative', or 'neutral', you may want to predict the number of stars it was given. This is a regression task and and it turns out we can use ULMFiT for that purpose as well.

We have observed that the regressor training often goes terribly wrong if the dataset is unbalanced. If there are some values that tend to dominate (e.g. reviews with 5 stars), the network will learn to output values between 4 and 6 regardless of input. If the imbalance isn't very large, you may do well with oversampling or adding weights to the loss function, but only up to the point where the data is better handled using anomaly detection techniques.

Example invocation:

```
python -m examples.ulmfit_tf_regressor \
          --train-tsv examples_data/hotels200.tsv \
          --data-column-name review \
          --gold-column-name rating \
          --model-weights-cp keras_weights/enwiki100_20epochs_35k_uncased \
          --model-type from_cp \
          --spm-model-file spm_model/enwiki100-uncased-sp35k.model \
          --max-seq-len 350 \
          --batch-size 32 \
          --num-epochs 12 \
          --loss-fn mse \
          --out-path ./hotel_regressor \
          --save-best
```

The `--loss-fn` can be set to `mse` (mean squared error) or `mae` (mean absolute error). You can also pass `--normalize-labels` to squeeze the response values into the range between 0 and 1, which often improves the optimizer's convergence.

After training the regressor, run the demo:

```
python -m examples.ulmfit_tf_regressor \
          --model-weights-cp hotel_regressor/best_checkpoint/best \
          --model-type from_cp \
          --spm-model-file spm_model/enwiki100-toks-sp35k-uncased.model \
          --max-seq-len 350 \
          --interactive

...

Paste a document to classify using a regressor: hotel no security stole laptop avoid like plague
Score: = [2.5436585]
Paste a document to classify using a regressor: excellent hotel stay wonderful staff
Score: = [6.9718037]
```



Compare this to a model trained with `--normalize-labels`:

```\
Paste a document to classify using a regressor: hotel no security stole laptop avoid like plague
Score: = [0.12547399]
Paste a document to classify using a regressor: excellent hotel stay wonderful staff
Score: = [1.8454201]
```



### 4.5. Sequence tagger (`ulmfit_tf_seqtagger.py`) with a custom training loop

A sequence tagging task involves marking each token as belonging to one of the classes. The typical examples here are Named Entity Recognition (NER) and Part-of-Speech (POS) tagging. Our example script uses input data with annotations on the word level (a small sample of sentences from the CONLL2013 dataset). These are provided as a JSONL file containing lists of tuples:

```a single training example from the JSONL file
[["Daniel", 5], ["Westlake", 6], [",", 0], ["46", 0], [",", 0], ["from", 0], ["Victoria", 1], [",", 0], ["made", 0], ["the", 0], ["first", 0], ["sucessful", 0], ["escape", 0], ["from", 0], ["Klongprem", 1], ["prison", 0], ["in", 0], ["the", 0], ["northern", 0], ["outskirts", 0], ["of", 0], ["the", 0], ["capital", 0], ["on", 0], ["Sunday", 0], ["night", 0], [".", 0]]
```

The numbers in the second element of each tuple are the token label defined in the labels map JSON file:

```
{"0": "O", "1": "B-LOC", "2": "I-LOC", "3": "B-ORG",
 "4": "I-ORG", "5": "B-PER", "6": "I-PER"}
```

The vast majority of publicly available datasets for such tasks contain annotations on the word level. However, our version of ULMFiT uses subword tokenization which is almost universally adopted in newer language models. This means that the labels need to be re-aligned to subwords. The `ulmfit_tf_tagger.py` script contains a simple function called `tokenize_and_align_labels` which does that for you. 

* Input word-level annotations from the training corpus

  ```
  Daniel         B-PER
  Westlake       I-PER
  ,                 O
  46                O
  from              O
  Victoria       B-LOC
  ...
  ```

* Annotations re-aligned to subwords:

  ```
  <s>               O
  ▁Daniel        B-PER
  ▁West          I-PER
  lake           I-PER
  ▁,                O
  ▁46               O
  ▁,                O
  ▁from             O
  ▁Victoria      B-LOC
  ...
  </s>              O
  ```

  

The `ulmfit_tf_tagger.py` scripts also illustrates how you can use the ULMFiT model in a custom training loop. Instead of building a Keras model and calling `model.fit`, we create a training function that uses `tf.GradientTape`. Pay close attention to how the AWD regularization is applied:

* if you are training on files from Tensorflow Hub - call the module's `apply_awd`serialized `tf.function` before each batch,
* if you are using Keras weights - call`apply_awd_eagerly` from `ulmfit_tf2.py` before each batch

```
def train_step(*, model, hub_object, loss_fn, optimizer, awd_off=None, x, y, step_info):
    if awd_off is not True:
        if hub_object is not None: hub_object.apply_awd(0.5)
        else: apply_awd_eagerly(model, 0.5)
    with tf.GradientTape() as tape:
        y_preds = model(x, training=True)
        loss_value = loss_fn(y_true=y, y_pred=y_preds)
        print(f"Step {step_info[0]}/{step_info[1]} | batch loss before applying gradients: {loss_value}")

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```



Example invocation (training):

```
python -m examples.ulmfit_tf_seqtagger \
          --train-jsonl examples_data/conll1000.jsonl \
          --label-map examples_data/tagger_labels.json \
          --spm-model-file spm_model/enwiki100-toks-sp35k-cased.model \
          --model-weights-cp keras_weights/enwiki100_20epochs_toks_35k_cased \
          --model-type from_cp \
          --batch-size 32 \
          --num-epochs 5 \
          --out-path ./conll_tagger
```

Example invocation (demo - note the side effect of coarse alignment of B-LOC with all subwords of the first word in the training corpus. A more careful approach would have been to align B-LOC with only the first subword. 'Melfordshire' could then be tagged as `_M/B-LOC, elf/I-LOC, ord/I-LOC, shire/I-LOC`):

```
python -m examples.ulmfit_tf_seqtagger \
          --label-map examples_data/tagger_labels.json \
          --model-weights-cp ./conll_tagger/tagger \
          --model-type from_cp \
          --spm-model-file spm_model/enwiki100-toks-sp35k-cased.model \
          --interactive

Write a sentence to tag: What is the weather in Melfordshire ?
<s>               O
▁What             O
▁is               O
▁the              O
▁weather          O
▁in               O
▁M             B-LOC
elf            B-LOC
ord            B-LOC
shire          B-LOC
▁                 O
?                 O
</s>              O
```





## 5. Pretraining your own language model from scratch (`pretraining_utils`)

This section describes how to use a raw text corpus to train an ULMFiT language model. As explained at the beginning of this document and on our TFHub page, we use FastAI to train encoder weights, which we then convert to a Tensorflow model with the [convert_fastai2keras.py](convert_fastai2keras.py) script.

Our data preparation methods are somewhat different from the approach taken by FastAI's authors described in [Training a text classifier](https://docs.fast.ai/tutorial.text.html#Training-a-text-classifier). In particular, our data is sentence-tokenized. We also dispense with special tokens for such as `\\xmaj` or `\\xxrep` and rely on sentencepiece tokenization entirely. For these reasons our training scripts skip all the preprocessing, tokenization, transforms and numericalization steps, and expect inputs to be provided in an already numericalized form. Some of these steps are factored out to external scripts instead, which you will find in the [pretraining_utils](pretraining_utils) directory and which are described below.

Obtaining source data and cleaning it up will require different techniques for each data source. Here we assume that you are already past this stage and **that you already have a reasonably clean raw corpus**. All you need to do is save it as three large plain text files (e.g. `train.txt`, `valid.txt` and `test.txt`) with one sentence per line. This is important since our scripts add BOS and EOS markers at the beginning and end of each line. As an alternative, you may want to train a language model on paragraphs or documents - in which each line in your text files will need to correspond to a paragraph or a document.



### 5.1. Basic cleanup and sentence tokenization (optional - `01_cleanup.py`)

If your text isn't already sentence-tokenized, you can split it into sentences in separate lines with the `01_cleanup.py` script. Some minor cleanup and preprocessing is also performed, in particular punctuation characters are separated from words by a whitespace (ex. `what?` -> `what ?`).

Example invocation:

```
python -m pretraining_utils.01_cleanup \
          --lang english \
          --input-text train_raw.txt \
          --out-path ./train_sents.txt
```



### 5.2. Build subword vocabulary (`02_build_spm.py`)

To avoid problems with out-of-vocabulary words, we recommend using a subword tokenizer such as **[Sentencepiece](https://github.com/google/sentencepiece)** (SPM). An important decision to take is how many subwords you want to have. Unless you are building something as big as GPT-3, it's probably safe to use ~30k subwords for English and maybe up to 50k for highly inflectional languages.

Those of you who are proficient in using sentencepiece from the command line can build SPM from Google's repo and then run `spm_train` manually. The advantage is that you get to tweak things like character coverage, but remember to set the control character indices as expected by FastAI (see section 3.1) and not to their default values. If you don't need this much flexibility, you can run our wrapper around SPM trainer in `02_build_spm.py`.

Example invocation:

```
python -m pretraining_utils.02_build_spm \
          --corpus-path train_sents.txt \
          --vocab-size 5000 \
          --model-prefix wittgenstein
```

This will produce two files: `wittgenstein-sp5k.model` and `wittgenstein-sp5k.vocab`. You really only need the first one for further steps.



### 5.3. Tokenize and numericalize your corpus (`03_encode_spm.py`) 

Now that you have the raw text and the sentencepiece tokenizer, you can convert text to tokens and their IDs:

- **Tokenization:** (observe the BOS/EOS markers!)
  
  ```
  ['<s>', '▁The', '▁lim', 'its', '▁of', '▁my', '▁language', '▁mean', '▁the', '▁lim', 'its', '▁of', '▁my', '▁world', '</s>']
  ```
  
- **Numericalization** (converting each subword to its ID in the dictionary):
  
  ```
  [2, 200, 2240, 1001, 20, 317, 2760, 286, 9, 2240, 1001, 20, 317, 669, 3]
  ```
  
  

Because it was important for us to have full control over numericalization (we didn't want to use FastAI's default functions), our training scripts require the corpus to be numericalized before being fed to FastAI data loaders. The `03_encode_spm.py` file will read a plain text file with the source corpus and save another text file with its numericalized version. It will also add BOS and EOS markers.

Example invocation:

```
python -m pretraining_utils.03_encode_spm \
          --corpus-path ./train_sents.txt \
          --model-path ./wittgenstein-sp5k.model \
          --spm-extra-options bos:eos \
          --output-format id \
          --save-path ./train_ids.txt
```



### 5.4. Training in FastAI

Congratulations! Now that your data is mangled, all you need to do is execute FastAI training. **Only do it on a CPU if your corpus is really tiny** as pretraining takes a significant amount of time. Before you run the training we advise that you study the argument descriptions (`--help` flag ) carefully, as there are several hyperparameters that will for sure need tweaking on your custom input. For large datasets (e.g. Wikipedia) in particular it will be useful to prepare a numericalized validation corpus and pass it via the `--pretokenized-valid` argument so that you can measure the perplexity metric after each epoch.

Example invocation:

```
python ./fastai_ulmfit_train.py \
       --pretokenized-train ./train_ids.txt \
       --min-seq-len 7 \
       --max-seq-len 80 \
       --batch-size 128 \
       --vocab-size 5000 \
       --num-epochs 20 \
       --save-path ./wittgenstein_lm \
       --exp-name wittg

...
epoch     train_loss  valid_loss  accuracy  perplexity  time    
0         6.454500    5.968120    0.094213  390.770355  02:40
1         5.841645    5.346557    0.181822  209.884445  02:35
2         5.404020    4.939891    0.204419  139.755066  03:07
3         5.085054    4.652817    0.228302  104.879990  03:35
4         4.877162    4.478850    0.241150  88.133247   03:38
....
```

The result is as FastAI model file called `wittg.pth` which will be saved in the `wittgensteir_lm` directory.



**5.5. Exporting to Tensorflow 2.0 and serializing the ExportableULMFiTRagged class**

WIP

### 5.6. Unsupervised fine-tuning a pretrained language model in FastAI

WIP

### 5.7. Perplexity evaluation

WIP



## 6. References and acknowledgements

NCBIR grant + author's contact details









