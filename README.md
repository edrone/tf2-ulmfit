# Language modelling scripts

This directory contains training scripts for recurrent language models. The [modelling_scripts](modelling_scripts) directory contains [Hubert's repo](https://github.com/hubertkarbowy/LanguageModellingScripts) as a submodule.



## How to train a new LSTM-based language model?

[WIP]

## How to run a next-token prediction demo?

The demo works similar to an autocompletion prompter, i.e. given "Drużyna Stanów Zjednoczonych zdobyła złoty medal na", the model should suggest that the next likely tokens are "mistrzostwach świata".

You will need:
* a pretrained model file (.hfd5) from s3://prod-edrone-ava/AVA-sandbox_resources/lm_recurrent
* a sentencepiece vocabulary model (.model) - from the same location

```
export HDF5=./wiki100_finetune-07.hdf5
export SPM_MODEL=./plwiki100-sp35k.model
cd modelling_scripts/lstm_with_wordpieces
python ./04_demo.py --pretrained-model ${HDF5} \
                    --spm-model-file ${SPM_MODEL} \
                    --add-bos \
                    --max-seq-len 80
```

## How to run a sequence tagging demo?

You will need:

* a pretrained model checkpoint files from s3://prod-edrone-ava/AVA-sandbox_resources/lm_recurrent/phuabc-ulmfit-tagger-demo
* a sentencepiece vocabulary model (plwiki100-sp35k.model) - from the same location

```
export TAGGER_CKPT=../resources/lm_recurrent/phuabc-ulmfit-tagger-demo/phu_1epoka
export WIKI100_SPM=../resources/lm_recurrent/phuabc-ulmfit-tagger-demo/plwiki100-sp35k.model
python ./ulmfit_tf_seqtagger.py \
       --model-weights-cp ${TAGGER_CKPT} \
       --spm-model-file ${WIKI100_SPM} \
       --interactive
```

There will be a couple of warnings about checkpoints being resolved to different objects - they are safe to ignore (we'll fix them later). Once the model loads, you will see its architecture and there will be a sentence input prompt:

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
* a pretrained model file (.hfd5) from s3://prod-edrone-ava/AVA-sandbox_resources/lm_recurrent
* a testset file (e.g. test.txt) - one sentence per line
* a sentencepiece vocabulary model (.model) - you will find it in the same location as the hdf5 file

Before running the evaluation script, convert the testset to token ids like this:

```
export TESTSET_FILE=test.txt
export SPM_MODEL=plwiki100-sp35k.model
cd modelling_scripts/lstm_with_wordpieces
python ./02b_encode_spm.py --corpus-path ${TESTSET_FILE} \
                           --model-path ${SPM_MODEL} \
                           --save-path /tmp/test_ids.txt \
                           --save-stats
```

The file `test_ids.txt` now contains tokenized text converted to token IDs. You can now run the evaluation script:

```
export TEST_IDS=/tmp/test_ids.txt
export HDF5=./wiki100_finetune-07.hdf5
cd ../..
python ./evaluate_ppl.py --corpus-path ${TEST_IDS} \
                         --model-type causal \
                         --pretrained-path ${HDF5} \
                         --tokenizer-file ${SPM_MODEL} \
                         --tokenizer-type spm \
                         --max-seq-len 80 \
                         --min-seq-len 10 \
                         --add-bos \
                         --add-eos \
                         --is-pretokenized
```
Note: perplexity evaluations are slow because you need to softmax over the entire vocabulary as many number of times as there are tokens. Make sure to run them on a sample of ~10k sentences, not more.

