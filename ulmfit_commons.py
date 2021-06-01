import os, subprocess
import numpy as np
from fastai.basics import *
from fastai.callback.all import *
from fastai.text.all import *

""" Various ULMFit / FastAI related utils """

def file_len(fname):
    """ Nothing beats wc -l """
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def read_labels(fname):
    label_map = open(fname, 'r', encoding='utf-8').readlines()
    label_map = {k:v.strip() for k,v in enumerate(label_map) if len(v)>0}
    return label_map

def lr_or_default(lr, learner_obj):
    if lr is not None:
        return lr
    else:
        lr_min, lr_steep = learner_obj.lr_find()
        print(f"LR finder results: min rate {lr_min}, rate at steepest gradient: {lr_steep}")
        return lr_steep

def get_fastai_tensors(args):
    """ Read pretokenized and numericalized corpora and return them as TensorText objects understood by
        the scantily documented FastAI's voodoo language model loaders.
    """
    L_tensors_train = L()
    L_tensors_valid = L()
    data_sources = [(args['pretokenized_train'], 'trainset', L_tensors_train)]
    if args.get('pretokenized_valid') is not None:
        data_sources.append((args['pretokenized_valid'], 'validset', L_tensors_valid))

    for datasource_path, datasource_name, L_tensors in data_sources:
        with open(datasource_path, 'r', encoding='utf-8') as f:
            print(f"Reading {datasource_name} from {datasource_path}")
            num_sents = file_len(datasource_path)
            cnt = 0
            for line in f:
                if cnt % 10000 == 0: print(f"Processing {datasource_name}: line {cnt} / {num_sents}...")
                tokens = TensorText(list(map(int, line.split())))
                if len(tokens) > args['min_seq_len']: L_tensors.append(tokens)
                cnt += 1
    return L_tensors_train, L_tensors_valid

def read_numericalize(*, input_file, sep='\t', spm_model_file, label_map=None, max_seq_len=None, fixed_seq_len=None,
                      x_col, y_col, sentence_tokenize=False, cut_off_final_token=False):
    import pandas as pd
    import sentencepiece as spm
    import nltk
    df = pd.read_csv(input_file, sep=sep)
    if label_map is not None:
        df[y_col] = df[y_col].astype(str)
        df[y_col].replace({v:k for k,v in label_map.items()}, inplace=True)
    if sentence_tokenize is True:
        df[x_col] = df[x_col].str.replace(' . ', '[SEP]', regex=False)
        df[x_col] = df[x_col].map(lambda t: nltk.sent_tokenize(t, language='polish'))\
                             .map(lambda t: "[SEP]".join(t))
    spmproc = spm.SentencePieceProcessor(spm_model_file)
    spmproc.set_encode_extra_options("bos:eos")
    x_data = spmproc.tokenize(df[x_col].tolist())
    if cut_off_final_token is True:
        x_data = [d[:-1] for d in x_data]
    if max_seq_len is not None:
        x_data = [d[:max_seq_len] for d in x_data]
    if fixed_seq_len is not None:
        x_data = [d + [1]*(fixed_seq_len - len(d)) for d in x_data]
    labels = df[y_col].tolist()
    return x_data, labels, df

def save_as_keras(*, state_dict, exp_name, save_path, spm_model_file):
    """
    Creates an ULMFit inference model using Keras layers and copies weights from FastAI's learner.model.state_dict() there.

    There are many explicit constants in this function, which is intentional. The numbers 400, 1152 and 3 layers refer
    to the paper's implementation of ULMFit in FastAI.

    """

    import tensorflow as tf
    from modelling_scripts.ulmfit_tf2 import tf2_ulmfit_encoder

    spm_args = {
        'spm_model_file': spm_model_file,
        'add_bos': True,
        'add_eos': True,
        'lumped_sents_separator': '[SEP]'
    }
    lm_num, encoder_num, outmask_num, spm_encoder_model = tf2_ulmfit_encoder(fixed_seq_len=None, spm_args=spm_args)

    lm_num.get_layer('ulmfit_embeds').set_weights([state_dict['0.encoder.weight'].cpu().numpy()])
    rnn_weights1 = [state_dict['0.rnns.0.module.weight_ih_l0'].cpu().numpy().T,
                    state_dict['0.rnns.0.weight_hh_l0_raw'].cpu().numpy().T,
                    state_dict['0.rnns.0.module.bias_ih_l0'].cpu().numpy()*2]
    rnn_weights2 = [state_dict['0.rnns.1.module.weight_ih_l0'].cpu().numpy().T,
                    state_dict['0.rnns.1.weight_hh_l0_raw'].cpu().numpy().T,
                    state_dict['0.rnns.1.module.bias_ih_l0'].cpu().numpy()*2]
    rnn_weights3 = [state_dict['0.rnns.2.module.weight_ih_l0'].cpu().numpy().T,
                    state_dict['0.rnns.2.weight_hh_l0_raw'].cpu().numpy().T,
                    state_dict['0.rnns.2.module.bias_ih_l0'].cpu().numpy()*2]

    lm_num.get_layer('AWD_RNN1').set_weights(rnn_weights1)
    lm_num.get_layer('AWD_RNN2').set_weights(rnn_weights2)
    lm_num.get_layer('AWD_RNN3').set_weights(rnn_weights3)
    lm_num.get_layer('lm_head_tied').set_weights([state_dict['1.decoder.bias'].cpu().numpy(),
                                                  state_dict['1.decoder.weight'].cpu().numpy()])
    lm_num.save_weights(os.path.join(save_path, exp_name))
    return lm_num, encoder_num, outmask_num, spm_encoder_model


