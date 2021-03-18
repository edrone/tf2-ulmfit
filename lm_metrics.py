import tensorflow as tf
import numpy as np
from corpus_feeder import tensor_shift

PAD_ID = 1

def calculate_ppl(*, restored_model, corpus_loader, is_pretokenized):
    """ Calculate perplexity """
    batch_generator = corpus_loader.next_batch_causal_lm(is_pretokenized=is_pretokenized, padding_direction='post')
    all_logprobs = []
    cnt = 0
    num_sents = 0
    for batch in batch_generator:
        if cnt % 10 == 0: print(f"Processing batch {cnt}")
        preds = restored_model(batch)
        batch_mask = tf.where(batch == PAD_ID, 0, 1) # put zeros on padding and ones on tokens
        batch_lengths = tf.reduce_sum(batch_mask, axis=1).numpy()
        shifted = tensor_shift(data=batch, positions=-1, axis=1, pad_fill=PAD_ID)
        indices = []
        for idx, single_length in enumerate(batch_lengths):
            # The line below generates a list of tuples (sentence_idx, token_idx, next_token_id)
            # For example, a tuple (0, 4, 5629) means: In the zeroeth sentence, the token *after* index 4 has id of 5629)
            next_token_tuples = list(zip([idx]*(single_length-1), np.arange(single_length-1), shifted[idx]))
            indices.extend(next_token_tuples)
        batch_logprobs = np.log(tf.gather_nd(preds, indices) + 1e-10) # 1e-10 is added to each softmax score so as not to accidentally encounter log(0.0)
        all_logprobs.extend(batch_logprobs)
        cnt += 1
        num_sents += len(batch)

    # Calculate final perplexity score: e^(-1/num_tokens * sum from 1 to num_tokens over all logprobabilities)
    ppl = (-np.sum(all_logprobs))/len(all_logprobs)
    ppl = np.exp(ppl)
    return ppl, num_sents

def calculate_pppl():
    """ Calculate pseudo-perplexity """
    raise NotImplementedError
