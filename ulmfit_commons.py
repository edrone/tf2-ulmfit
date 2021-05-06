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

def save_as_keras(*, state_dict, exp_name, save_path, awd_weights, fixed_seq_len, spm_model_file):
    """
    Creates an ULMFit inference model using Keras layers and copies weights from FastAI's learner.model.state_dict() there.

    There are many explicit constants in this function, which is intentional. The numbers 400, 1152 and 3 layers refer
    to the paper's implementation of ULMFit in FastAI.

    """

    import tensorflow as tf
    from modelling_scripts.ulmfit_tf2_heads import ulmfit_sequence_tagger
    from modelling_scripts.ulmfit_tf2 import tf2_ulmfit_encoder, TiedDense

    spm_args = {'spm_model_file': spm_model_file,
                'add_bos': True,
                'add_eos': True,
                'lumped_sents_separator': '[SEP]'
    }

    if awd_weights == 'on':
        lm_num, encoder_num, outmask_num, spm_encoder_model = tf2_ulmfit_encoder(fixed_seq_len=fixed_seq_len, use_awd=True, spm_args=spm_args, flatten_ragged_outputs=True)
        rnn_layer1 = 'AWD_RNN1'
        rnn_layer2 = 'AWD_RNN2'
        rnn_layer3 = 'AWD_RNN3'
    elif awd_weights == 'off':
        lm_num, encoder_num, outmask_num, spm_encoder_model = tf2_ulmfit_encoder(fixed_seq_len=fixed_seq_len, use_awd=False, spm_args=spm_args, flatten_ragged_outputs=True)
        rnn_layer1 = 'Plain_LSTM1'
        rnn_layer2 = 'Plain_LSTM2'
        rnn_layer3 = 'Plain_LSTM3'
    else:
        raise ValueError(f"Unknown awd_weights argument {awd_weights}!")
    lm_num.summary()
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

    if awd_weights == 'on':
        rnn_weights1.append(state_dict['0.rnns.0.weight_hh_l0_raw'].cpu().numpy().T)
        # rnn_weights1.append(np.array(False))
        rnn_weights2.append(state_dict['0.rnns.1.weight_hh_l0_raw'].cpu().numpy().T)
        # rnn_weights2.append(np.array(False))
        rnn_weights3.append(state_dict['0.rnns.2.weight_hh_l0_raw'].cpu().numpy().T)
        # rnn_weights3.append(np.array(False))

    lm_num.get_layer(rnn_layer1).set_weights(rnn_weights1)
    lm_num.get_layer(rnn_layer2).set_weights(rnn_weights2)
    lm_num.get_layer(rnn_layer3).set_weights(rnn_weights3)
    lm_num.get_layer('lm_head_tied').set_weights([state_dict['1.decoder.bias'].cpu().numpy(),
                                                  state_dict['1.decoder.weight'].cpu().numpy()])
    lm_num.save_weights(os.path.join(save_path, exp_name))
    return lm_num, encoder_num, outmask_num, spm_encoder_model
