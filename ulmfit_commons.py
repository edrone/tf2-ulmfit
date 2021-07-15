import os
import re
import subprocess
import sys

import nltk
import pandas as pd
import sentencepiece as spm

# from ulmfit_tf2 import AWDCallback, LRFinder


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

def read_numericalize(*, input_file, sep='\t', spm_model_file, label_map=None, max_seq_len=None, fixed_seq_len=None,
                      x_col, y_col, sentence_tokenize=False, cut_off_final_token=False):
    df = pd.read_csv(input_file, sep=sep)
    if label_map is not None:
        df[y_col] = df[y_col].astype(str)
        df[y_col].replace({v:k for k,v in label_map.items()}, inplace=True)
    if sentence_tokenize is True:
        df[x_col] = df[x_col].str.replace(' . ', '[SEP]', regex=False)
        df[x_col] = df[x_col].map(lambda t: nltk.sent_tokenize(t, language='polish')) \
                             .map(lambda t: [re.sub(r"(\w)([,!\?])", "\\1 \\2 ", sent) for sent in t]) \
                             .map(lambda t: "[SEP]".join(t))
    spmproc = spm.SentencePieceProcessor(spm_model_file)
    spmproc.set_encode_extra_options("bos:eos")
    x_data = spmproc.tokenize(df[x_col].tolist())
    if cut_off_final_token is True:
        x_data = [d[:-1] for d in x_data]
    if max_seq_len is not None:
        x_data = [d[:max_seq_len] for d in x_data]
    if fixed_seq_len is not None:
        x_data = [d[:fixed_seq_len] for d in x_data]
        x_data = [d + [1]*(fixed_seq_len - len(d)) for d in x_data]
    labels = df[y_col].tolist()
    return x_data, labels, df

def check_unbounded_training(fixed_seq_len, max_seq_len):
    if not any([fixed_seq_len, max_seq_len]):
        print("Warning: you have requested training with an unspecified sequence length. " \
             "This script will not truncate any sequence, but you should make sure that " \
             "all your training examples are reasonably long. You should be fine if your " \
             "training set is split into sentences, but DO make sure that none of them " \
             "runs into thousands of tokens or you will get out-of-memory errors.\n\n")
        sure = "?"
        while sure not in ['y', 'Y', 'n', 'N']:
            sure = input("Are you sure you want to continue? (y/n) ")
        if sure in ['n', 'N']:
            sys.exit(1)

def prepare_keras_callbacks(*, args, model, hub_object,
                            monitor_metric='val_sparse_categorical_accuracy'):
    """Build a list of Keras callbacks according to command-line parameters parsed into `args`."""
    import tensorflow as tf  # tensorflow global import conflicts with fastai
    from ulmfit_tf2 import AWDCallback, LRFinder
    callbacks = []
    if not args.get('awd_off'):
        callbacks.append(AWDCallback(model_object=model if hub_object is None else None,
                                     hub_object=hub_object))
    if args.get('lr_finder'):
        max_steps = args['lr_finder']
        lr_finder_cb = LRFinder(max_steps=max_steps)
        callbacks.append(lr_finder_cb)
    if args.get('save_best') is True:
        best_dir = os.path.join(args['out_path'], 'best_checkpoint')
        os.makedirs(best_dir, exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(best_dir, 'best'),
                                            monitor=monitor_metric,
                                            save_best_only=True,
                                            save_weights_only=True))
    if args.get('tensorboard'):
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='tboard_logs', update_freq='batch'))
    return callbacks

def print_training_info(*, args, x_train, y_train):
    """
    Print training information to stdout

    :param dict args:       Arguments dictionary (see the argparse fields in individual scripts)
    :param x_train:         A tensor of training examples
    :param y_train:         A tensor of training labels

    """
    num_steps = (x_train.shape[0] // args['batch_size']) * args['num_epochs']
    print(f"************************ TRAINING INFO ***************************\n" \
          f"Shapes - sequence inputs: {x_train.shape}, labels: {y_train.shape}\n" \
          f"Batch size: {args['batch_size']}, Epochs: {args['num_epochs']}, \n" \
          f"Steps per epoch: {x_train.shape[0] // args['batch_size']} \n" \
          f"Total steps: {num_steps}\n" \
          f"AWD after each batch: {'off' if args.get('awd_off') is True else 'on'}\n" \
          f"******************************************************************")
