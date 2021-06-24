"""
Runs an interactive demo of a pretrained ULMFiT language model.

Given a sentence beginning, it will try to predict the next tokens.
Now also works with serialized SavedModels!

python -m modelling_scripts.lstm_with_wordpieces.04_demo \
          --pretrained-model $PRETRAINED_MODELS/wiki-pl-100-sp50k-cased/keras_weights/plwiki100_20epochs_50k_cased \
          --model-type from_cp \
          --spm-model-file $PRETRAINED_MODELS/wiki-pl-100-sp50k-cased/spm_model/plwiki100-sp50k-cased.model \
          --add-bos
"""
import argparse
import heapq
import logging
import readline

import numpy as np
import sentencepiece as spm
import tensorflow as tf

from ulmfit_tf2_heads import ulmfit_rnn_encoder_native, ulmfit_rnn_encoder_hub

UNK_ID=0; PAD_ID=1; BOS_ID=2; EOS_ID=3
logging.basicConfig(level=logging.INFO)

def get_spm_extra_opts(args):
    extra_opts=[]
    if args.get('add_bos') is True:
        extra_opts.append("bos")
    if args.get('add_eos') is True:
        extra_opts.append("eos")
    return ":".join(extra_opts)

def predict_next_n_pieces(model, spmproc, sent, args):
    piece_ids = spmproc.encode_as_ids(sent)
    cnt = 0
    for i in range(args['max_lookahead_tokens']):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(piece_ids)
        print(f"Encoded as {len(piece_ids)} pieces: {[spmproc.id_to_piece(piece) for piece in piece_ids]}")
        # if args.get('max_seq_len') is not None: # model saved with a fixed sequence length, so adding padding
        #     last_token_idx = len(encoded) - 1 if args['padding_direction'] == 'post' \
        #         else args['max_seq_len'] - 1
        #     x_hat = tf.keras.preprocessing.sequence.pad_sequences(
        #         np.array([piece_ids]), \
        #         maxlen=args['max_seq_len'],
        #         value=PAD_ID,
        #         padding=args['padding_direction']
        #     )
        #else:
        x_hat = tf.ragged.constant([piece_ids])
        last_token_idx = len(piece_ids) - 1
        y_hat = model.predict(x_hat)

        next_k_piece_ids = heapq.nlargest(args['beam_width'], \
                                          range(0, spmproc.vocab_size()), \
                                          np.array(y_hat[0, last_token_idx]).take)
        print(next_k_piece_ids)
        next_k_piece_probs = np.array(y_hat[0, last_token_idx]).take([next_k_piece_ids]).tolist()[0]
        print(next_k_piece_probs)
        next_k_pieces = [spmproc.id_to_piece(p) for p in next_k_piece_ids]
        print(f"Candidate next pieces: {list(zip(next_k_pieces, next_k_piece_probs))}")
        cnt += 1
        if next_k_piece_ids[0] == EOS_ID or cnt >= args['max_lookahead_tokens']:
            break
        piece_ids.append(next_k_piece_ids[0])

def main(args):
    spm_model_args = {
        'spm_model_file': args['spm_model_file'],
        'add_bos': args.get('add_bos') or False,
        'add_eos': args.get('add_eos') or False,
        'lumped_sents_separator': '[SEP]'
    }
    spmproc = spm.SentencePieceProcessor(args['spm_model_file'])
    spmproc.SetEncodeExtraOptions(get_spm_extra_opts(args))
    logging.info(f"SPM processor detected vocab size of {spmproc.vocab_size}. First 10 tokens:")    
    logging.info(str([spmproc.id_to_piece(i) for i in range(10)]))
    logging.info("Restoring a pretrained language model")    
    if args['model_type'] == 'from_cp':
        lm_num = ulmfit_rnn_encoder_native(pretrained_weights=args['pretrained_model'],
                                           spm_model_args=spm_model_args,
                                           also_return_spm_encoder=False,
                                           return_lm_head=True)
    elif args['model_type'] == 'from_hub':
        _, lm_num, _ = ulmfit_rnn_encoder_hub(pretrained_weights=args['pretrained_model'],
                                              spm_model_args=spm_model_args,
                                              also_return_spm_encoder=False,
                                              return_lm_head=True)
    else:
        raise NotImplementedError("Unsupported model source {args['model_type']}!")
    lm_num.summary()
    readline.parse_and_bind('set editing-mode vi')
    while True:
        sent = input("Write a sentence to complete: ")
        predict_next_n_pieces(lm_num, spmproc, sent, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model", required=True, help="Path to Keras weights for ULMFiT or the SavedModel (ragged version)")
    parser.add_argument("--model-type", required=True, default='from_cp', choices=['from_cp', 'from_hub'], help="Input model format")
    parser.add_argument("--spm-model-file", required=True, help="SPM .model file")
    parser.add_argument("--add-bos", required=False, action='store_true', help="Will add <s> tokens")
    parser.add_argument("--add-eos", required=False, action='store_true', help="Will add </s> tokens. Should generally not be added for prediction/demo")
    parser.add_argument("--padding-direction", choices=['pre', 'post'], default='post', help="Pre or post padding (for LM training 'post' seems better than 'pre'). Irrelevant if input is a ragged tensor.")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam search not implemented, defaulting to 1 (greedy search)")
    parser.add_argument("--max-lookahead-tokens", type=int, default=3, help="Maximum number of tokens to generate after input sequence. The 'decoder' will stop when it hits </s>.")
    argz = parser.parse_args()
    main(vars(argz))
