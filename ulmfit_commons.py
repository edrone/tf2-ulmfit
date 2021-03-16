import subprocess
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

        
    
