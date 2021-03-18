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