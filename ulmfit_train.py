import os, argparse
import numpy as np
import subprocess
import pickle

from fastai.basics import *
from fastai.callback.all import *
from fastai.text.all import *
from ulmfit_commons import lr_or_default, get_fastai_tensors

"""
Pre-train ULMFit on already tokenized and numericalized corpora.

This script roughly follows the original FastAI's tutorial (https://docs.fast.ai/tutorial.wikitext.html#Model)
but we get rid of *ALL* Jeremy Howard's preprocessing quirks. In fact, we don't trust his preprocessing code
so much that we prefer to delegate all the vocabulary building, tokenization and numericalization to
a command-line invocation of SentencePiece. The only real thing we keep from FastAI is the training itself.

"""

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

    We basically try not to overfit on the new dataset. How to do this is really art, not science,
    so in the future we'll have to figure out experimentally what works best for AVA. For now
    I tried to regularize the model as much as possible by:

    1. Freezing the recurrent layers during the first epoch with the same learning rate as
       the pretrained model.
    2. Unfreezing all the layers and training them with a much lower LR.

    This link gives some ideas: https://humboldt-wi.github.io/blog/research/information_systems_1819/group4_ulmfit/
    but authors use the old scheduler (slanted triangular rates), so the code below is somewhat adapted.
    """
    print(f"Will resume pretraining from {args['pretrained_model']}")
    print(f"Freezing all recurrent layers, leaving trainable LM head tied to embeddings")
    learner_obj.load(args['pretrained_model'])
    learner_obj.model[0].rnns.requires_grad_(False)
    lr = lr_or_default(args['pretrain_lr'], learner_obj)
    learner_obj.fit_one_cycle(1, lr, moms=(0.8, 0.7, 0.8), div=10)
    learner_obj.unfreeze()
    lr = lr_or_default(args['finetune_lr'], learner_obj)
    learner_obj.fit_one_cycle(args['num_epochs']-1, slice(lr/100, lr), pct_start=0.3, div=10)
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
    # this looks like an update specific to the Wikitext-2 corpus? The tutorial doesn't say anything about this, though.
    config.update({'input_p': 0.6,
                   'output_p': 0.4,
                   'weight_p': 0.5,
                   'embed_p': 0.1,
                   'hidden_p': 0.2})
    ulmfit_model = get_language_model(AWD_LSTM, args['vocab_size'], config=config) # produces a 3-layer LSTM as per the ULMFit paper
    opt_func = partial(Adam, wd=0.1, eps=1e-7)
    callbacks = [MixedPrecision(), GradientClip(0.1)] + rnn_cbs(alpha=2, beta=1)
    learner_obj = Learner(data_loaders, ulmfit_model, loss_func=CrossEntropyLossFlat(), opt_func=opt_func, \
                          cbs=callbacks, metrics=[accuracy, Perplexity()])
    print(learner_obj.model)
    learner_obj.model_dir = '.'
    if args.get('pretrained_model') is not None:
        learner_obj = _run_finetuning(learner_obj, args)
    else:
        learner_obj = _run_pretraining(learner_obj, args)
    print("Saving the ULMFit model in FastAI format ...")
    learner_obj.save(os.path.join(args['save_path'], args['exp_name'])) # .pth will be added automatically
    print("Saving the ULMFit model's weights into a pickle ...")
    pickle.dump(learner_obj.model.state_dict(), open(os.path.join(args['save_path'], f"{args['exp_name']}_state_dict.p"), 'wb'))
    if args.get('export_to_tf'):
        print("Saving the ULMFit model as Keras Sequential object")
        save_as_keras(learner_obj.model.state_dict(), args)
    return learner_obj

def save_as_keras(state_dict, args):
    """
    Creates an ULMFit inference model using Keras layers and copies weights from FastAI's learner.model.state_dict() there.

    There are many explicit constants in this function, which is intentional. The numbers 400, 1152 and 3 layers refer
    to the paper's implementation of ULMFit in FastAI.

    """

    import tensorflow as tf
    kmodel = tf.keras.models.Sequential()
    kmodel.add(tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True)) # ragged tensors = variable input length!
    kmodel.add(tf.keras.layers.Embedding(args['vocab_size'], 400, name='ulmfit_embedz'))
    kmodel.add(tf.keras.layers.LSTM(1152, return_sequences=True, name='ulmfit_lstm1'))
    kmodel.add(tf.keras.layers.LSTM(1152, return_sequences=True, name='ulmfit_lstm2'))
    kmodel.add(tf.keras.layers.LSTM(400, return_sequences=True, name='ulmfit_lstm3'))
    kmodel.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(args['vocab_size'], activation='linear'), name='lm_head')) # originally: linear

    kmodel.get_layer('ulmfit_embedz').set_weights([state_dict['0.encoder.weight'].cpu().numpy()])
    kmodel.get_layer('ulmfit_lstm1').set_weights([state_dict['0.rnns.0.module.weight_ih_l0'].cpu().numpy().T,
                                                  state_dict['0.rnns.0.weight_hh_l0_raw'].cpu().numpy().T,
                                                  state_dict['0.rnns.0.module.bias_ih_l0'].cpu().numpy()*2])
    kmodel.get_layer('ulmfit_lstm2').set_weights([state_dict['0.rnns.1.module.weight_ih_l0'].cpu().numpy().T,
                                                  state_dict['0.rnns.1.weight_hh_l0_raw'].cpu().numpy().T,
                                                  state_dict['0.rnns.1.module.bias_ih_l0'].cpu().numpy()*2])
    kmodel.get_layer('ulmfit_lstm3').set_weights([state_dict['0.rnns.2.module.weight_ih_l0'].cpu().numpy().T,
                                                  state_dict['0.rnns.2.weight_hh_l0_raw'].cpu().numpy().T,
                                                  state_dict['0.rnns.2.module.bias_ih_l0'].cpu().numpy()*2])
    kmodel.get_layer('lm_head').set_weights([state_dict['1.decoder.weight'].cpu().numpy().T,
                                             state_dict['1.decoder.bias'].cpu().numpy()])
    kmodel.save(os.path.join(args['save_path'], f"{args['exp_name']}_tf.hdf5"))

if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument("--pretokenized-train", required=False, help="Path to a pretokenized and numericalized training corpus. " \
                      "Make sure you have <s> and </s> tokens there as needed because ULMFit will concatenate everything " \
                      "into one big stream!")
    argz.add_argument("--pretokenized-valid", required=False, help="Path to a pretokenized and numericalized validation corpus. " \
                      "Same tokenization rules apply as for the training corpus.")
    argz.add_argument("--pretrained-model", required=False, help="Path to a pretrained FastAI/PyTorch model OR a state dict. " \
                      "Use this as input to Keras conversion (state dict) or if you want to resume pretraining (pth model).")
    argz.add_argument("--min-seq-len", default=10, type=int, help="Minimal sentence length in the original corpus")
    argz.add_argument("--max-seq-len", default=40, type=int, help="Maximal sequence length in a training batch. This is the same as BPTT.")
    argz.add_argument("--batch-size", default=64, type=int, help="Batch size")
    argz.add_argument("--vocab-size", required=True, type=int, help="Vocabulary size")
    argz.add_argument("--num-epochs", default=1, type=int, help="Number of epochs to train for")
    argz.add_argument("--pretrain-lr", required=False, type=float, help="Learning rate value for the one cycle policy optimizer. "\
                                                                      "At pretraining: the optimizer will use it for all layers. "\
                                                                      "At finetuning: this lr will be used for one epoch on unfrozen LM head only") # 5e-3
    argz.add_argument("--finetune-lr", required=False, type=float, help="Learning rate value for the one cycle policy optimizer. "\
                                                                      "Only used for finetuning starting from the second epoch.") # 5e-4
    argz.add_argument("--save-path", required=True, help="Path where the outputs will be saved")
    argz.add_argument("--exp-name", required=True, help="Experiment name")
    argz.add_argument("--export-to-numpy", action='store_true', help="Will save an additional pickle containing layer weights "\
                                                                     "as Numpy arrays. These can be loaded into a TensorFlow inference model.")
    argz.add_argument("--export-to-tf", action='store_true', help="Will save an additional hdf5 file containing Keras-loadable "\
                                                                  "ULMFit model with LM head.")

    argz = vars(argz.parse_args())
    if argz.get('pretrained_model') is not None and argz.get('export_to_tf') is True and argz.get('pretokenized_train') is None:
        state_dict = pickle.load(open(argz['pretrained_model'], 'rb'))
        save_as_keras(state_dict, argz)
    elif argz.get('pretokenized_train') is None:
        print("Please provide a path to pretokenized and numericalized corpus")
        print(argz)
        exit(1)
    else:
        main(argz)
