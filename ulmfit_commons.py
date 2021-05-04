import os, subprocess
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

def save_as_keras(state_dict, exp_name, save_path, awd_weights, spm_model_file):
    """
    Creates an ULMFit inference model using Keras layers and copies weights from FastAI's learner.model.state_dict() there.

    There are many explicit constants in this function, which is intentional. The numbers 400, 1152 and 3 layers refer
    to the paper's implementation of ULMFit in FastAI.

    """

    import tensorflow as tf
    from modelling_scripts.ulmfit_tf2_heads import ulmfit_sequence_tagger, ulmfit_baseline_tagger
    from modelling_scripts.ulmfit_tf2 import tf2_ulmfit_encoder, TiedDense

    if awd_weights == 'on':
        kmodel, _, _, _ = tf2_ulmfit_encoder(fixed_seq_len=None, spm_model_file=spm_model_file)
        rnn_layer1 = 'AWD_RNN1'
        rnn_layer2 = 'AWD_RNN2'
        rnn_layer3 = 'AWD_RNN3'
    elif awd_weights == 'off':
        kmodel = ulmfit_baseline_tagger(fixed_seq_len=768, spm_model_file=None)
        rnn_layer1 = 'Plain_LSTM1'
        rnn_layer2 = 'Plain_LSTM2'
        rnn_layer3 = 'Plain_LSTM3'
        kmodel.pop() # tagger head
        kmodel.pop() # dropout
        kmodel.add(tf.keras.layers.TimeDistributed(TiedDense(reference_layer=kmodel.layers[0], activation='softmax'), name='lm_head_tied'))
        kmodel.add(tf.keras.layers.Dropout(0.05))
    else:
        raise ValueError(f"Unknown awd_weights argument {awd_weights}!")
    # kmodel = tf.keras.models.Sequential()
    # kmodel.add(tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True)) # ragged tensors = variable input length!
    # kmodel.add(tf.keras.layers.Embedding(args['vocab_size'], 400, name='ulmfit_embedz'))
    # kmodel.add(tf.keras.layers.LSTM(1152, return_sequences=True, name='ulmfit_lstm1'))
    # kmodel.add(tf.keras.layers.LSTM(1152, return_sequences=True, name='ulmfit_lstm2'))
    # kmodel.add(tf.keras.layers.LSTM(400, return_sequences=True, name='ulmfit_lstm3'))
    # kmodel.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(args['vocab_size'], activation='linear'), name='lm_head')) # originally: linear

    kmodel.get_layer('ulmfit_embeds').set_weights([state_dict['0.encoder.weight'].cpu().numpy()])
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
        rnn_weights2.append(state_dict['0.rnns.1.weight_hh_l0_raw'].cpu().numpy().T)
        rnn_weights3.append(state_dict['0.rnns.2.weight_hh_l0_raw'].cpu().numpy().T)

    kmodel.get_layer(rnn_layer1).set_weights(rnn_weights1)
    kmodel.get_layer(rnn_layer2).set_weights(rnn_weights2)
    kmodel.get_layer(rnn_layer3).set_weights(rnn_weights3)
    kmodel.get_layer('lm_head_tied').set_weights([state_dict['1.decoder.bias'].cpu().numpy(),
                                                  state_dict['1.decoder.weight'].cpu().numpy()])
    kmodel.save_weights(os.path.join(save_path, exp_name))


