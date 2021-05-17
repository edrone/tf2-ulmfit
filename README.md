# Language modelling scripts

This directory contains training scripts for recurrent language models. The [modelling_scripts](modelling_scripts) directory contains [Hubert's repo](https://github.com/hubertkarbowy/LanguageModellingScripts) as a submodule.


## How to train a new ULMFiT language model?

We still do this in FastAI and then convert the weights to a format usable with TensorFlow. To train a new language model, you will need:

1. **A cleaned-up and sentence-tokenized corpus**. You basically want to have three bing text files (train, valid, test) with one sentence per line. Wikipedia is always a good starting point as a source of data and you can use a tool such as WikiExtractor to create dumps and generate these text files. Sentence tokenization can then be done with [this script](modelling_scripts/lstm_with_wordpieces/01_cleanup.py) (it also does some basic cleanup). Note: we suggest that you do not use the ready-made WikiText-103 dataset available from Salesforce as it is preprocessed in a way which you may not necessarily like.
2. **A SentencePiece vocabulary model** trained on your corpus. Use [this script](modelling_scripts/lstm_with_wordpieces/02_build_spm.py) to produce .model and .vocab files (there is an example invocation in the docstring - have a look). You will need to decide how big you want your vocabulary to be. If you have no idea, a rough indication for a model trained on Wikipedia is 35k tokens for English and 50k tokens for inflectional languages.
3. **A numericalized version of your corpus**. We do not tokenize and numericalize the corpus during training because FastAI's methods do a lot of intensive preprocessing. This means that any input for inference has to be preprocessed identically. We do away with these inventions and instead provide the Learner object with an already numericalized corpus. The [02b_encode_spm.py](modelling_scripts/lstm_with_wordpieces/02b_encode_spm.py) script does that for you - again have a look at the docstring for a sample invocation.

Now you can run the training (note: `--max-seq-len` is actually BPTT. It does not truncate any sequences.). Do make some experiments with the learning rate on a small dataset before training on the proper corpus. The LR of 0.005 should generally give decent results but fairly high perplexity. We found that 0.01 is maybe a little too fast at the beginning, but fine for English and it does eventually converge in the second step of the 1-cycle schedule for other languages as well.  

```
export TRAINSET=train_ids.txt
export VALIDSET=valid_ids.txt
export SPMMODEL=plwiki100-sp50k-cased.model
python ./ulmfit_train.py --pretokenized_train ${TRAINSET} \
                         --pretokenized-valid ${VALIDSET} \
                         --spm-model-file ${SPMMODEL} \
                         --min-seq-len 8 \
                         --max-seq-len 70 \
                         --batch-size 128 \
                         --vocab-size 50000 \
                         --num-epochs 20 \
                         --pretrain-lr 0.01 \
                         --save-path ./output_directory \
                         --exp-name my_experiment
```
After 20 epochs you will have a file called `my_experiment.pth` in the `output_directory`. 

## How to convert a model trained in FastAI to a Tensorflow version?

Execute the following:

```
export PTH_FILE=/tmp/plwiki100_20epochs_50k_cased.pth
export SPMMODEL=plwiki100-sp50k-cased.model
python ./ulmfit_convert_fastai2keras.py \
         --pretrained-model ${PTH_FILE} \
         --out-path ./converted \
         --spm-model-file ${SPMMODEL}
```

Tensorflow actually uses two formats:
1. just the model weights. To use it in your own project, you will need access to the original Python code that builds the `tf.keras.models.Model` object and its layers.
2. the SavedModel. The model is converted to a Tensorflow graph and serialized together with its weights. You do not need access to the original Python code to use it.

The conversion script generates both formats. It also copies the SPM model and the original FastAI .pth file. When the script completes, you should expect the `converted` directory to contain the following structure:

```
├── fastai_model
│   └── plwiki100_20epochs_50k_cased.pth
├── keras_weights
│   ├── checkpoint
│   ├── plwiki100_20epochs_50k_cased.data-00000-of-00001
│   └── plwiki100_20epochs_50k_cased.index
├── saved_model
│   ├── assets
│   │   └── plwiki100-sp50k-cased.model
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
└── spm_model
    ├── plwiki100-sp50k-cased.model
    └── plwiki100-sp50k-cased.vocab

```

## How to run a next-token prediction demo?

The demo works similar to an autocompletion prompter, i.e. given "Drużyna Stanów Zjednoczonych zdobyła złoty medal na", the model should suggest that the next likely tokens are "mistrzostwach świata".

You will need:
* a pretrained from s3://prod-edrone-ava/AVA-sandbox_resources/lm_recurrent
* a sentencepiece vocabulary model (.model) - from the same location

The 04_demo.py script can read both Keras weights as well as the SavedModel binaries, but you have to tell it which format you are passing via the `--model-type` parameter (either `from_hub` or `from_cp`). Below is an example for the SavedModel format:

```
export SPM_MODEL=./plwiki100-sp35k.model
export PRETRAINED=/home/hkarbowy/code/ava-sandbox/resources/lm_recurrent/wiki-pl-100/plwiki100_20epochs_50k_uncased/saved_model/
export MODEL_TYPE=from_hub

python -m modelling_scripts.lstm_with_wordpieces.04_demo \
          --pretrained-model ${PRETRAINED} \
          --model-type {MODEL_TYPE} \
          --add-bos
```

## How to train a sequence tagger?

[WIP]

## How to run a sequence tagging demo?

You will need:

* a pretrained model checkpoint files from s3://prod-edrone-ava/AVA-sandbox_resources/lm_recurrent/phuabc-ulmfit-tagger-demo
* a sentencepiece vocabulary model (plwiki100-sp35k.model) - from the same location

```
export TAGGER_CKPT=../resources/lm_recurrent/phuabc-ulmfit-tagger-demo/nowy_tagger
export WIKI100_SPM=../resources/lm_recurrent/phuabc-ulmfit-tagger-demo/plwiki100-sp35k.model
python ./ulmfit_tf_seqtagger.py \
       --model-weights-cp ${TAGGER_CKPT} \
       --model-type from_cp \
       --spm-model-file ${WIKI100_SPM} \
       --interactive
```

There will be a couple of warnings about sundry things - they are safe to ignore (we'll fix them later). Once the model loads, you will see its architecture and there will be a sentence input prompt:

```
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
_________________________________________________________________
time_distributed (TimeDistri (None, None, 3)           1203      
=================================================================
Total params: 34,263,667
Trainable params: 34,263,667
Non-trainable params: 0
_________________________________________________________________
Write a sentence to tag:
```
You can now try out some typical sentences alluding to descriptions of products typically found in a hardware shop:

```
Write a sentence to tag: Jak masz chęć to se wkręć śrubkę o średnicy numer pięć.
<s>               O
▁Jak              O
▁masz             O
▁chęć             O
▁to               O
▁se               O
▁w                O
krę               O
ć                 O
▁śru              O
b                 O
kę                O
▁o                O
▁średnicy       B-N
▁numer            O
▁pięć             O
.                 O
</s>              O
```

## How to train a sequence tagger?

[WIP]

## How to evaluate a pretrained model's perplexity?

You will need:
* a pretrained FastAI file (.pth) from s3://prod-edrone-ava/AVA-sandbox_resources/lm_recurrent
* a numericalized testset file - one sentence per line. You can use the 02b_encode_spm.py script on the same .model file that was used for training.

Let's say the file `/tmp/test_ids.txt`  contains tokenized text converted to token IDs. You can now run the evaluation script:

```
export TEST_IDS=/tmp/test_ids.txt
export PRETRAINED=./converted/fastai_model/plwiki100_20epochs_50k_cased.pth
python ./ulmfit_ppl.py --pretokenized-test ${TEST_IDS} \
                       --pretrained-model ${PRETRAINED} \
                       --min-seq-len 10 \
                       --max-seq-len 100 \
                       --batch-size 128 \
                       --vocab-size 50000
```
Note: perplexity evaluations are slow because you need to softmax over the entire vocabulary as many number of times as there are tokens. Make sure to run them on a sample of ~10k sentences, not more.