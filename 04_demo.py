import argparse
import logging
import pickle
import tensorflow as tf
import numpy as np
import heapq
import readline
import sentencepiece as spm

"""
Runs an interactive demo of a pretrained language model.

Given a sentence beginning, it will try to predict the next tokens. Usage:

python ./04_demo.py \
        --pretrained-model ./model3-40.hdf5 \
        --spm-model-file ./plato-sp5k.model \
        --max-seq-len 40 \
        --add-bos
"""

# todo: migrate spmto tensorflow_text?
UNK_ID=0; PAD_ID=1; BOS_ID=2; EOS_ID=3
logging.basicConfig(level=logging.INFO)

def get_spm_extra_opts(args):
    extra_opts=[]
    if args.get('add_bos') is True:
        extra_opts.append("bos")
    if args.get('add_eos') is True:
        extra_opts.append("eos")
    return ":".join(extra_opts)

def prepare_sequences(spmproc, args):
    x_sequences = []
    cnt = 0
    with open(args['pretokenized_text'], 'r', encoding='utf-8') as f:
        for line in f:
            if cnt % 1000: logging.info(f"Tokenizing line {cnt}")
            pieces = spmproc.encode_as_ids(line)
            if len(pieces) < args['min_seq_len']: continue
            if len(pieces) > args['max_seq_len']:
                pieces = pieces[0:args['max_seq_len']]
                if args['add_eos'] is True: pieces[-1] = 3 # fixme: this is hardcoded - not good!
            x_sequences.append(pieces)
            cnt += 1

    logging.info("Tokenization completed. First 10 sentences:")
    for i in range(10):
        pieces = [spmproc.id_to_piece(x) for x in x_sequences[i]]
        logging.info(str(pieces))
    x_sequences = tf.keras.preprocessing.sequence.pad_sequences(x_sequences, \
                                                                padding=args['padding_direction'], \
                                                                maxlen=args['max_seq_len'], \
                                                                value=PAD_ID)
    return x_sequences

def predict_next_n_pieces(model, spmproc, sent, args):
    encoded = spmproc.encode_as_pieces(sent)
    cnt = 0
    for i in range(args['max_lookahead_tokens']):
        print(f"Encoded as {len(encoded)} pieces: {encoded}")
        last_token_idx = len(encoded) - 1 if args['padding_direction'] == 'post' \
                         else args['max_seq_len'] - 1
        x_hat = tf.keras.preprocessing.sequence.pad_sequences(
                   np.array([spmproc.encode_as_ids(spmproc.decode_pieces(encoded))]), \
                   maxlen=args['max_seq_len'],
                   value=PAD_ID,
                   padding=args['padding_direction']
                   )
        y_hat = model.predict(x_hat)
        next_k_piece_ids = heapq.nlargest(args['beam_width'], \
                                          range(0, spmproc.vocab_size()), \
                                          y_hat[0, last_token_idx].take)
        next_k_piece_probs = y_hat[0, last_token_idx, next_k_piece_ids]
        next_k_pieces = [spmproc.id_to_piece(p) for p in next_k_piece_ids]
        print(f"Candidate next pieces: {list(zip(next_k_pieces, next_k_piece_probs))}")
        cnt += 1
        if next_k_piece_ids == EOS_ID or cnt >= args['max_lookahead_tokens']:
            break
        encoded.append(next_k_pieces[0])

def main(args):
    spmproc = spm.SentencePieceProcessor(args['spm_model_file'])
    spmproc.SetEncodeExtraOptions(get_spm_extra_opts(args))
    logging.info(f"SPM processor detected vocab size of {spmproc.vocab_size}. First 10 tokens:")    
    logging.info(str([spmproc.id_to_piece(i) for i in range(10)]))
    logging.info("Restoring a pretrained language model")    
    simple_model = tf.keras.models.load_model(args['pretrained_model'])
    simple_model.summary()
    readline.parse_and_bind('set editing-mode vi')
    while True:
        sent = input("Write a sentence to complete: ")
        predict_next_n_pieces(simple_model, spmproc, sent, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model", required=True, help="Path to a .hdf5 file containing the pretrained language model")
    parser.add_argument("--spm-model-file", required=True, help="SPM .model file")
    parser.add_argument("--add-bos", required=False, action='store_true', help="Will add <s> tokens")
    parser.add_argument("--add-eos", required=False, action='store_true', help="Will add </s> tokens. Should generally not be added for prediction/demo")
    parser.add_argument("--max-seq-len", required=True, type=int, help="Maximum number of wordpiece tokens in a sequence")
    parser.add_argument("--padding-direction", choices=['pre', 'post'], default='post', help="Pre or post padding (for LM training 'post' seems better than 'pre')")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam search not implemented, defaulting to 1 (greedy search)")
    parser.add_argument("--max-lookahead-tokens", type=int, default=3, help="Maximum number of tokens to generate after input sequence. The 'decoder' will stop when it hits </s>.")
    argz = parser.parse_args()
    main(vars(argz))
