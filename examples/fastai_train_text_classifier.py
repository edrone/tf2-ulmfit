"""
Fine-tune an ULMFiT text classifier from:

    1. A pretrained language model
    2. Custom data stored in a tsv file

The data is numericalized using SentencePiece directly (rather than via FastAI's functions).
We dispense with "special tokens" (xxmaj, xxfld etc) and pre/post tokenization rules as well.
This implies that the pretrained language model must have been trained using the same subwords
dictionary and no default pre/post tokenization rules.

"""
import argparse
import os
import re
from collections import OrderedDict
from functools import partial
from operator import attrgetter

import numpy as np
import pandas as pd
from fastai.text.all import *
from sklearn.metrics import classification_report

from ulmfit_commons import read_labels, read_numericalize


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


def read_tsv_and_numericalize(*, tsv_file, args, also_return_df=False):
    label_map = read_labels(args['label_map'])
    x_data, y_data, df = read_numericalize(input_file=tsv_file,
                                           spm_model_file=args['spm_model_file'],
                                           label_map=label_map,
                                           fixed_seq_len = args.get('fixed_seq_len'),
                                           max_seq_len = args.get('fixed_seq_len'),
                                           x_col=args['data_column_name'],
                                           y_col=args['gold_column_name'],
                                           sentence_tokenize=True,
                                           cut_off_final_token=False)
    if also_return_df is True:
        return x_data, y_data, label_map, df
    else:
        return x_data, y_data, label_map


def make_fastai_learner(*, args, x_train, y_train, x_test, y_test, label_map, splits):
    x_data = [TensorText(k) for k in x_train] + [TensorText(k) for k in x_test]
    y_data = [TensorCategory(c) for c in y_train] + [TensorCategory(c) for c in y_test]

    df = pd.DataFrame.from_dict({'numericalized': x_data, 'labels': y_data})
    ds = Datasets(df, [[attrgetter('numericalized')], [attrgetter('labels')]], splits=splits)
    dls = ds.dataloaders(bs=args['batch_size'], shuffle=True)

    clas_config = awd_qrnn_clas_config if args.get('qrnn') else awd_lstm_clas_config
    clas_config.update({'n_layers': args['num_hidden_layers']})
    fastai_text_classifier = get_text_classifier(arch=AWD_QRNN if args.get('qrnn') else AWD_LSTM,
						 config=clas_config,
                                                 vocab_sz=args['vocab_size'],
                                                 n_class=len(label_map),
                                                 seq_len=args['fixed_seq_len'],
                                                 drop_mult=0.5) # RNN encoder + denses with BatchNorm
    opt_func = partial(Adam, wd=0.1, eps=1e-7)
    callbacks = [MixedPrecision(),
                 GradientClip(0.1),
                 SaveModelCallback(fname=(args.get('exp_name') or "exp")+'_fastai_ckpt', every_epoch=True) \
                ] + rnn_cbs(alpha=2, beta=1)
    learner_obj = Learner(dls, fastai_text_classifier, loss_func=CrossEntropyLossFlat(),
                          opt_func=opt_func, cbs=callbacks, metrics=[accuracy])
    return learner_obj, fastai_text_classifier, dls


def evaluate(args):
    x_test, y_test, label_map, df = read_tsv_and_numericalize(tsv_file=args['test_tsv'], args=args, also_return_df=True)
    learner_obj, fastai_text_classifier, dls = make_fastai_learner(args=args,
                                                                   x_train=[], y_train=[],
                                                                   x_test=x_test, y_test=y_test,
                                                                   label_map=label_map, splits=None)
    learner_obj.model_dir = os.path.dirname(os.path.abspath(args['pretrained_model']))
    learner_obj.load(os.path.splitext(os.path.basename(args['pretrained_model']))[0])
    print(f"Restored FastAI text classifier from {args['pretrained_model']}")
    learner_obj.model.eval()
    logits, _, _ = learner_obj.model(TensorText(x_test))
    softmaxed = F.softmax(logits, dim=1)
    y_probs, y_preds = torch.max(softmaxed, dim=1)
    y_probs = y_probs.detach().to('cpu').numpy().tolist()
    y_preds = y_preds.detach().to('cpu').numpy().tolist()
    y_preds_labels = [label_map[l] for l in y_preds]
    print(classification_report(y_test, y_preds, target_names=list(label_map.values())))
    if args.get('save_path') is not None:
        df2 = pd.DataFrame.from_dict({'nltext': df[args['data_column_name']].tolist(),
                                      'gold': [label_map[l] for l in df[args['gold_column_name']].tolist()],
                                      'preds': y_preds_labels,
                                      'probs': y_probs,
                                      'softmaxes': softmaxed.detach().to('cpu').numpy().tolist(),
                                     })
        df2['result'] = np.where(df2['gold'] == df2['preds'], 'SUCCESS', 'FAIL')
        df2.to_csv(args['save_path'], sep='\t', index=None)


def train(args):
    x_train, y_train, label_map = read_tsv_and_numericalize(tsv_file=args['train_tsv'], args=args)
    if args.get('test_tsv'):
        train_len = len(x_train)
        x_test, y_test, _ = read_tsv_and_numericalize(tsv_file=args['test_tsv'], args=args)
        test_len = len(x_test)
        splits = [range(0, train_len), range(train_len, train_len+test_len)]
    else:
        x_test = []
        y_test = []
        splits = None

    learner_obj, fastai_text_classifier, dls = make_fastai_learner(args=args,
                                                                   x_train=x_train, y_train=y_train,
                                                                   x_test=x_test, y_test=y_test,
                                                                   label_map=label_map,
                                                                   splits=splits)
    if args.get('pretrained_model') is not None:
        print(f"Restoring a pretrained encoder from {args['pretrained_model']}")
        restore_encoder(pth_file=args['pretrained_model'], text_classifier=fastai_text_classifier)
    else:
        print("Warning: Training the classifier from a randomly initialized model. Are you sure " \
              "you don't want to use any pretrained weights?")
    print(learner_obj.model)
    print(dls.one_batch())
    learner_obj.model_dir = '..'
    if args.get('classifier_lr') is not None:
        learner_obj.fit_one_cycle(args['num_epochs'], args['classifier_lr'])
    else:
        learner_obj.fine_tune(args['num_epochs'])
    print("Saving the ULMFit model in FastAI format ...")
    os.makedirs(args['save_path'], exist_ok=True)
    learner_obj.save(os.path.join(args['save_path'], args['exp_name'])) # .pth will be added automatically
    return learner_obj


if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument("--train-tsv", required=False, help="Path to a training corpus. The script will handle"
                                                          "numericalization via the spm model.")
    argz.add_argument("--test-tsv", required=False, help="Path to a testing corpus")
    argz.add_argument("--spm-model-file", required=True, help="Path to SPM model")
    argz.add_argument("--pretrained-model", required=False, help="Path to a pretrained FastAI/PyTorch model. ")
    argz.add_argument("--fixed-seq-len", type=int, required=True, help="Maximal sequence length.")
    argz.add_argument("--label-map", required=True, help="Path to a labels file (one label per line)")
    argz.add_argument("--batch-size", default=64, type=int, help="Batch size")
    argz.add_argument("--vocab-size", required=True, type=int, help="Vocabulary size")
    argz.add_argument("--num-epochs", required=False, type=int, help="Number of epochs to train for")
    argz.add_argument("--classifier-lr", required=False, type=float, help="Learning rate value for the 1- cycle policy optimizer.")  # 5e-4
    argz.add_argument("--qrnn", required=False, action='store_true', help="Use QRNN instead of LSTM")
    argz.add_argument("--num-hidden-layers", required=False, type=int, help="Number of hidden layers in the encoder")
    argz.add_argument("--save-path", required=False, help="Path where the outputs will be saved")
    argz.add_argument("--exp-name", required=False, help="Experiment name")
    argz.add_argument('--data-column-name', default='sentence', help="Name of the column containing X data")
    argz.add_argument('--gold-column-name', default='target', help="Name of the gold column in the tsv file")

    argz = vars(argz.parse_args())
    if argz.get('num_hidden_layers') is None:
        argz['num_hidden_layers'] = 4 if argz.get('qrnn') else 3
    if argz.get('train_tsv') is not None:
        assert argz.get('num_epochs') is not None, "Please provide the number of epochs in training mode"
        assert argz.get('save_path') is not None, "Please provide the output path and experiment name"
        assert argz.get('exp_name') is not None, "Please provide the output path and experiment name"
        train(argz)
    else:
        evaluate(argz)
