"""
Pre-train ULMFit on already tokenized and numericalized corpora.

This script roughly follows the original FastAI's tutorial (https://docs.fast.ai/tutorial.wikitext.html#Model)
but we dispense with their preprocessing transforms. Instead, we provide an already numericalized corpus
to the trainer object.
"""
import argparse
import os
from functools import partial

from fastai.basics import *
from fastai.callback.all import *
from fastai.text.all import *
from fastai_lm_utils import get_fastai_tensors, lr_or_default


def _run_pretraining(learner_obj, args):
    """
    Runs pre-training of a new ULMFit model from scratch
    """
    learner_obj.fit_one_cycle(args['num_epochs'],
                              args.get('pretrain_lr') or 5e-3,
                              moms=(0.8, 0.7, 0.8),
                              div=10)
    learner_obj.model.reset()
    return learner_obj

def _run_finetuning(learner_obj, args):
    """
    Finetune an existing ULMFit model.

    We basically try not to cause catastrophic forgetting of the pretrained model.
    One approach might be to:

    1. Freeze the recurrent layers during the first epoch with the same learning rate as
       the pretrained model.
    2. Unfreeze all the layers and train them with a much lower LR.

    This link gives some ideas: https://humboldt-wi.github.io/blog/research/information_systems_1819/group4_ulmfit/
    but authors use the old version of ULMFiT's scheduler (slanted triangular rates), so the code below
    is not 100% faithful to the blog post.
    """
    print(f"Will resume pretraining from {args['pretrained_model']}")
    print(f"Freezing all recurrent layers, leaving trainable LM head tied to embeddings")
    learner_obj.load(args['pretrained_model'])
    learner_obj.model[0].rnns.requires_grad_(False)
    lr = lr_or_default(args['pretrain_lr'], learner_obj)
    learner_obj.fit_one_cycle(1, lr, moms=(0.8, 0.7, 0.8), div=50)
    learner_obj.unfreeze()
    lr = lr_or_default(args['finetune_lr'], learner_obj)
    learner_obj.fit_one_cycle(args['num_epochs']-1, lr_max=lr, pct_start=0.25,
                              div=25.0, div_final=100000.0, wd=0.1)
    return learner_obj

def main(args):
    L_tensors_train, L_tensors_valid = get_fastai_tensors(args)
    if L_tensors_valid == []:
        splits = None
    else:
        splits = [range(0, len(L_tensors_train)), range(len(L_tensors_train), len(L_tensors_train)+len(L_tensors_valid))]
    datasets = Datasets(L_tensors_train+L_tensors_valid, [add(0)],
                        splits=splits, dl_type=LMDataLoader) # no idea what FastAI's idiom for "identity" is, so faking it with add(0)
    print("Instantiating a DataLoaders object with automatic sequence shifter. This may take some time...")
    data_loaders = datasets.dataloaders(bs=args['batch_size'],
                                        seq_len=args['max_seq_len']) # to access a batch, use data_loaders.one_batch().
                                                                     # The data_loaders object also has .train and .valid fields if needed.

    ############# The actual FastAI training happens below ############

    config = awd_lstm_lm_config.copy()
    config.update({'input_p': 0.6,
                   'output_p': 0.4,
                   'weight_p': 0.5,
                   'embed_p': 0.1,
                   'hidden_p': 0.2})
    ulmfit_model = get_language_model(AWD_LSTM, args['vocab_size'], config=config) # produces a 3-layer LSTM as per the ULMFit paper
    opt_func = partial(Adam, wd=0.1, eps=1e-7)
    callbacks = [MixedPrecision(),
                 GradientClip(0.1),
                 SaveModelCallback(fname=args['exp_name']+'_fastai_ckpt', every_epoch=True) \
                ] + rnn_cbs(alpha=2, beta=1)
    learner_obj = Learner(data_loaders, ulmfit_model, loss_func=CrossEntropyLossFlat(), opt_func=opt_func, \
                          cbs=callbacks, metrics=[accuracy, Perplexity()])
    print(learner_obj.model)
    learner_obj.model_dir = '.'
    if args.get('pretrained_model') is not None:
        learner_obj = _run_finetuning(learner_obj, args)
    else:
        learner_obj = _run_pretraining(learner_obj, args)
    print("Saving the ULMFit model in FastAI format ...")
    os.makedirs(args['save_path'], exist_ok=True)
    learner_obj.save(os.path.join(args['save_path'], args['exp_name'])) # .pth will be added automatically

if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument("--pretokenized-train", required=False, help="Path to a pretokenized and numericalized training corpus. "
                      "Make sure you have <s> and </s> tokens there as needed because ULMFit will concatenate everything "
                      "into one big stream!")
    argz.add_argument("--pretokenized-valid", required=False, help="Path to a pretokenized and numericalized validation corpus. "
                      "Same tokenization rules apply as for the training corpus.")
    argz.add_argument("--pretrained-model", required=False, help="Path to a pretrained FastAI model. Use this for unsupervised "
                                                                 "finetuning")
    argz.add_argument("--min-seq-len", default=10, type=int, help="Minimal sentence length in the original corpus")
    argz.add_argument("--max-seq-len", default=80, type=int, help="Maximal sequence length in a training batch. This is the same as BPTT.")
    argz.add_argument("--batch-size", default=128, type=int, help="Batch size")
    argz.add_argument("--vocab-size", required=True, type=int, help="Vocabulary size")
    argz.add_argument("--num-epochs", default=1, type=int, help="Number of epochs to train for")
    argz.add_argument("--pretrain-lr", required=False, type=float, help="Learning rate value for the one cycle policy optimizer. "
                      "At pretraining: the optimizer will use it for all layers. "
                      "At finetuning: this lr will be used for one epoch on unfrozen LM head only") # 5e-3
    argz.add_argument("--finetune-lr", required=False, type=float, help="Learning rate value for the one cycle policy optimizer. "
                                                                      "Only used for finetuning starting from the second epoch.") # 5e-4
    argz.add_argument("--save-path", required=True, help="Path where the outputs will be saved")
    argz.add_argument("--exp-name", required=True, help="Experiment name")
    argz = vars(argz.parse_args())
    main(argz)
