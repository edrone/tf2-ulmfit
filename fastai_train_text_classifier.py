import os, re, argparse
import numpy as np
import subprocess
import pickle
from fastai.basics import *
from fastai.callback.all import *
from fastai.text.all import *
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from collections import OrderedDict
from torch.utils.data import TensorDataset
from ulmfit_commons import lr_or_default, get_fastai_tensors, read_labels
from ulmfit_text_classifier import read_numericalize

"""
Train an ULMFiT text classifier from a numericalized corpus

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

def restore_encoder(*, pth_file, text_classifier):
    encoder = get_model(text_classifier)[0].module
    wgts = torch.load(pth_file, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    renamed_keys = OrderedDict()
    for k, v in wgts['model'].items():
        if not k.startswith('0'):
            continue
        else:
            renamed_keys[re.sub('^0.', '', k)] = v
    encoder.load_state_dict(renamed_keys)
    print("Encoder was restored")

def main(args):
    label_map = read_labels(args['label_map'])
    x_train, y_train, _ = read_numericalize(input_file=args['train_tsv'],
                                                 spm_model_file=args['spm_model_file'],
                                                 label_map=label_map,
                                                 fixed_seq_len = args.get('fixed_seq_len'),
                                                 x_col=args['data_column_name'],
                                                 y_col=args['gold_column_name'],
                                                 sentence_tokenize=True,
                                                 cut_off_final_token=True)
    if args.get('test_tsv'):
        train_len = x_train.shape[0]
        x_test, y_test, _ = read_numericalize(input_file=args['test_tsv'],
                                              spm_model_file=args['spm_model_file'],
                                              label_map=label_map,
                                              fixed_seq_len = args.get('fixed_seq_len'),
                                              x_col=args['data_column_name'],
                                              y_col=args['gold_column_name'],
                                              sentence_tokenize=True,
                                              cut_off_final_token=True)
        test_len = x_test.shape[0]
        splits = [range(0, train_len), range(train_len, train_len+test_len)]
    else:
        x_test = tf.constant([])
        y_test = np.array([])
        splits = None
    x_data = [TensorText(k) for k in x_train.numpy().tolist()] + [TensorText(k) for k in x_test.numpy().tolist()]
    y_data = [TensorCategory(c) for c in y_train.tolist()] + [TensorCategory(c) for c in y_test.tolist()]

    df = pd.DataFrame.from_dict({'numericalized': x_data, 'labels': y_data})
    ds = Datasets(df, [[attrgetter('numericalized')], [attrgetter('labels')]], splits=splits)
    data_loaders = ds.dataloaders(bs=args['batch_size'], seq_len=args.get('fixed_seq_len'), shuffle=True)

    ############# The actual FastAI training happens below ############

    # this looks like an update specific to the Wikitext-2 corpus? The tutorial doesn't say anything about this, though.
    fastai_text_classifier = get_text_classifier(arch=AWD_LSTM,
                                                 vocab_sz=args['vocab_size'],
                                                 n_class=len(label_map),
                                                 seq_len=args['fixed_seq_len'],
                                                 drop_mult=0.5) # RNN encoder + denses with BatchNorm
    opt_func = partial(Adam, wd=0.1, eps=1e-7)
    callbacks = [MixedPrecision(),
                 GradientClip(0.1),
                 SaveModelCallback(fname=args['exp_name']+'_fastai_ckpt', every_epoch=True) \
                ] + rnn_cbs(alpha=2, beta=1)
    restore_encoder(pth_file=args['pretrained_model'], text_classifier=fastai_text_classifier)
    learner_obj = Learner(data_loaders, fastai_text_classifier, loss_func=CrossEntropyLossFlat(),
                          opt_func=opt_func, cbs=callbacks, metrics=[accuracy])
    print(learner_obj.model)
    learner_obj.model_dir = '.'
    learner_obj.fit_one_cycle(12, 0.01)
    exit(0)

    learner_obj = _run_finetuning(learner_obj, args)

    print("Saving the ULMFit model in FastAI format ...")
    os.makedirs(args['save_path'], exist_ok=True)
    learner_obj.save(os.path.join(args['save_path'], args['exp_name'])) # .pth will be added automatically
    # print("Saving the ULMFit model's weights into a pickle ...")
    # pickle.dump(learner_obj.model.state_dict(), open(os.path.join(args['save_path'], f"{args['exp_name']}_state_dict.p"), 'wb'))
    if args.get('export_to_tf2'):
        print("Saving the ULMFit model to a Keras checkpoint...")
        save_as_keras(state_dict=learner_obj.model.state_dict(),
                      exp_name=args['exp_name'],
                      save_path=args['save_path'],
                      awd_weights='on',
                      fixed_seq_len=None,
                      spm_model_file=args['spm_model_file'])
        print("Done")
    return learner_obj

if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument("--train-tsv", required=False, help="Path to a training corpus. The script will handle numericalization via the spm model.")
    argz.add_argument("--valid-tsv", required=False, help="Path to a validation / testing corpus")
    argz.add_argument("--spm-model-file", required=True, help="Path to SPM model")
    argz.add_argument("--pretrained-model", required=False, help="Path to a pretrained FastAI/PyTorch model. ")
    argz.add_argument("--min-seq-len", default=10, type=int, help="Minimal sentence length in the original corpus")
    argz.add_argument("--fixed-seq-len", type=int, required=True, help="Maximal sequence length.")
    argz.add_argument("--label-map", required=True, help="Path to a labels file (one label per line)")
    argz.add_argument("--batch-size", default=64, type=int, help="Batch size")
    argz.add_argument("--vocab-size", required=True, type=int, help="Vocabulary size")
    argz.add_argument("--num-epochs", default=1, type=int, help="Number of epochs to train for")
    argz.add_argument("--classifier-lr", required=False, type=float, help="Learning rate value for the one cycle policy optimizer. "\
                                                                      "Only used for finetuning starting from the second epoch.") # 5e-4
    argz.add_argument("--save-path", required=True, help="Path where the outputs will be saved")
    argz.add_argument("--exp-name", required=True, help="Experiment name")
    argz.add_argument('--data-column-name', default='sentence', help="Name of the column containing X data")
    argz.add_argument('--gold-column-name', default='target', help="Name of the gold column in the tsv file")


    argz = vars(argz.parse_args())
    main(argz)
