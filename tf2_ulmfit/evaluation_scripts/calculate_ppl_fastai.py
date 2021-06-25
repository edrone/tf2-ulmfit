"""
Calculate the perplexity of a pretrained ULMFit language model the FastAI way

FastAI appears to lump all sentences in the entire corpus into one long string and shift
them one token at a time. We keep the original script here since it's faster (though different)
than our implementation in TF.

"""
import argparse
import os
from functools import partial

import torch
import torch.nn.functional as F
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.rnn import rnn_cbs
from fastai.callback.training import GradientClip
from fastai.data.core import Datasets
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat
from fastai.optimizer import Adam
from fastai.text.data import LMDataLoader
from fastai.text.models import AWD_LSTM, get_language_model
from tensorflow import add

from tf2_ulmfit.fastai_lm_utils import get_fastai_tensors


def main(args):
    L_tensors_train, L_tensors_valid = get_fastai_tensors(args)
    splits = [range(0, len(L_tensors_train)), range(len(L_tensors_train), len(L_tensors_train)+len(L_tensors_valid))]
    datasets = Datasets(L_tensors_train+L_tensors_valid, [add(0)], dl_type=LMDataLoader)
    print("Instantiating a DataLoaders object with automatic sequence shifter. This may take some time...")
    # to access a batch, use data_loaders.one_batch()
    # The data_loaders object also has .train and .valid fields if needed.
    data_loaders = datasets.dataloaders(bs=args['batch_size'],
                                        seq_len=args['max_seq_len'])

    # Build the learner object
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ulmfit_model = get_language_model(AWD_LSTM, args['vocab_size'])  # produces a 3-layer LSTM as per the ULMFit paper
    opt_func = partial(Adam, wd=0.1, eps=1e-7)
    callbacks = [MixedPrecision(), GradientClip(0.1)] + rnn_cbs(alpha=2, beta=1)
    learner_obj = Learner(data_loaders,
                          ulmfit_model,
                          loss_func=CrossEntropyLossFlat(),
                          opt_func=opt_func,
                          cbs=[],
                          metrics=[])
    print(learner_obj.model)

    # Restore the pretrained model (.pth file)

    learner_obj.model_dir = os.path.dirname(os.path.abspath(args['pretrained_model']))
    fname = os.path.basename(os.path.abspath(args['pretrained_model']))
    fname = fname.strip(".pth")
    learner_obj.load(fname, device=device)
    learner_obj.model.to(device)
    num_batches = data_loaders.train.n_batches
    bgen = data_loaders.train.create_batches(range(num_batches*args['batch_size']))
    ce_losses = [] # cross entropy losses per batch

    # The actual perplexity calculation loop
    bnum = 0
    for x_sequences, targets in bgen: # x = token ids, y = x shifted by 1 !!! THE MODEL IS STATEFUL !!!
        print(f"Processing batch {bnum}/{num_batches}")
        with torch.no_grad():
            outputs = learner_obj.model(x_sequences.to(device))[0].to(device)
        for i in range(len(outputs)):
            ce_losses.append(F.cross_entropy(outputs[i, :, :].to(device), targets[i]).to(device))
        bnum += 1
    ppl = torch.exp(torch.sum(torch.tensor(ce_losses)) / len(ce_losses))
    print(f"Perplexity = {ppl} (on {len(ce_losses)*args['batch_size']} sequences (stateful!)")


if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument("--pretokenized-test", required=False, help="Path to a pretokenized and "
                      "numericalized validation corpus. Same tokenization rules apply as for the training corpus.")
    argz.add_argument("--pretrained-model", required=True, help="Path to a pretrained FastAI/PyTorch model (.pth)")
    argz.add_argument("--min-seq-len", default=10, type=int, help="Minimal sentence length in the original corpus")
    argz.add_argument("--max-seq-len", default=70, type=int, help="Maximal sequence length in a batch")
    argz.add_argument("--batch-size", default=64, type=int, help="Batch size")
    argz.add_argument("--vocab-size", required=True, type=int, help="Vocabulary size")

    argz = vars(argz.parse_args())
    argz['pretokenized_train'] = argz['pretokenized_test']
    main(argz)
