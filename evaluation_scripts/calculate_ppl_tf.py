"""
Evaluate perplexity of a causal language model in Tensorflow.
"""
import argparse

import numpy as np
import tensorflow as tf

from corpus_feeder import LMCorpusLoader
from corpus_feeder import tensor_shift
from ulmfit_tf2_heads import ulmfit_rnn_encoder_native

PAD_ID = 1


def calculate_ppl(*, restored_model, corpus_loader, needs_softmaxing=False):
    """ Calculate perplexity """
    batch_generator = corpus_loader.next_batch()
    all_logprobs = []
    cnt = 0
    num_sents = 0
    for batch in batch_generator:
        if cnt % 10 == 0: print(f"Processing batch {cnt}")
        preds = restored_model(batch)
        if needs_softmaxing:
            preds = tf.nn.softmax(preds, axis=2)
        batch_mask = tf.where(batch == PAD_ID, 0, 1) # put zeros on padding and ones on tokens
        batch_lengths = tf.reduce_sum(batch_mask, axis=1).numpy()
        shifted = tensor_shift(data=batch, positions=-1, axis=1, pad_fill=PAD_ID)
        indices = []
        for idx, single_length in enumerate(batch_lengths):
            # The line below generates a list of tuples (sentence_idx, token_idx, next_token_id)
            # For example, a tuple (0, 4, 5629) means: In the zeroeth sentence, the token *after*
            # index 4 has id of 5629)
            next_token_tuples = list(zip([idx]*(single_length-1), np.arange(single_length-1), shifted[idx]))
            # tmp_indices = next_token_tuples
            # cross_entropy_loss = -np.sum(np.log(tf.gather_nd(preds, tmp_indices)))/single_length
            # print(f"CE loss = {cross_entropy_loss}")
            indices.extend(next_token_tuples)
        batch_logprobs = np.log(tf.gather_nd(preds, indices) + 1e-10)  # 1e-10 is added to each softmax score
        all_logprobs.extend(batch_logprobs)                            # for numerical stability
        cnt += 1
        num_sents += len(batch)

    # Calculate final perplexity score: e^(-1/num_tokens * sum from 1 to num_tokens over all logprobabilities)
    ppl = (-np.sum(all_logprobs))/len(all_logprobs)
    ppl = np.exp(ppl)
    return ppl, num_sents


def main(args):
    if args.get('spm_model_file') is not None:
        spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': True, 'add_eos': True,
                    'lumped_sents_separator': '[SEP]'}
    else:
        spm_args = None
    lm_head, spm_encoder = ulmfit_rnn_encoder_native(pretrained_weights=args['pretrained_path'],
                                                     fixed_seq_len=args.get('max_seq_len'),
                                                     spm_model_args=spm_args,
                                                     also_return_spm_encoder=True,
                                                     return_lm_head=True)
    lm_head.summary()
    corpus_loader = LMCorpusLoader(corpus_path=args['corpus_path'],
                                   batch_size=args['batch_size'],
                                   min_seq_len=args['min_seq_len'],
                                   max_seq_len=args['max_seq_len'],
                                   spm_encoder=spm_encoder,
                                   is_numericalized=args['is_numericalized'],
                                   padding_direction='post',)
    ppl_score, num_sents = calculate_ppl(restored_model=lm_head,
                                         corpus_loader=corpus_loader,
                                         needs_softmaxing=args.get('lm_head_needs_softmaxing'))
    print(f"Perplexity = {ppl_score} (on {num_sents} sentences)")


if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument("--corpus-path", required=True, help="Path to a corpus file."
                      "One line = one sentence = one training/test example")
    argz.add_argument("--pretrained-path", required=True, help="Path to a directory containing a pretrained LM.")
    argz.add_argument("--spm-model-file", required=False, help="Path to a sentencepiece model file")
    argz.add_argument("--is-numericalized", required=False, action='store_true', help="Set this flag if your corpus"
                      "file is already numericalized")
    argz.add_argument("--max-seq-len", default=100, type=int, help="Maximum sequence length")
    argz.add_argument("--min-seq-len", default=1, type=int, help="Minimum sequence length")
    argz.add_argument("--add-bos", action='store_true', help="Whether to add <s> tokens to each sentence")
    argz.add_argument("--add-eos", action='store_true', help="Whether to add </s> tokens to each sentence")
    argz.add_argument("--lm-head-needs-softmaxing", action='store_true', help="If set, the LM head has linear"
                      "activation and scores need to be softmaxed before evaluating PPL. Set this flag if you"
                      "want to evaluate the .pth model from FastAI")
    argz.add_argument("--batch-size", type=int, default=64, help="Default batch size for predictions")
    argz = vars(argz.parse_args())

    if argz.get('is_numericalized') is None:
        assert argz.get('spm_model_file') is not None, "If your corpus isn't converted to token IDs, please" \
                                                       "provide a path to the SPM model file."
    main(argz)