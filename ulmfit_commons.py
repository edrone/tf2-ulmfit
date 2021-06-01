import subprocess
import sys

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
