import argparse
import os
import readline

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

from ulmfit_commons import read_labels, read_numericalize, check_unbounded_training, print_training_info, \
    prepare_keras_callbacks
from ulmfit_tf2 import STLRSchedule, OneCycleScheduler, PredictionProgressCallback, SPMNumericalizer, LRFinder
from ulmfit_tf2_heads import ulmfit_document_classifier


def read_tsv_and_numericalize(*, tsv_file, args, also_return_df=False):
    label_map = read_labels(args['label_map'])
    x_data, y_data, df = read_numericalize(input_file=tsv_file,
                                           spm_model_file=args['spm_model_file'],
                                           label_map=label_map,
                                           max_seq_len=args.get('max_seq_len'),
                                           fixed_seq_len=args.get('fixed_seq_len'),
                                           x_col=args['data_column_name'],
                                           y_col=args['gold_column_name'],
                                           sentence_tokenize=True,
                                           cut_off_final_token=False)
    if args.get('fixed_seq_len') is not None:
        x_data = tf.constant(x_data, dtype=tf.int32)
    else:
        x_data = tf.ragged.constant(x_data, dtype=tf.int32)
    y_data = tf.constant(y_data, dtype=tf.int32)
    if also_return_df is True:
        return x_data, y_data, label_map, df
    else:
        return x_data, y_data, label_map


def interactive_demo(args):
    spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': True, 'add_eos': True,
                'lumped_sents_separator': '[SEP]'}
    label_map = read_labels(args['label_map'])
    spm_encoder = SPMNumericalizer(spm_path=args['spm_model_file'],
                                   add_bos=True,
                                   add_eos=True,
                                   fixed_seq_len=args.get('fixed_seq_len'))
    model, _ = ulmfit_document_classifier(model_type=args['model_type'],
                                          pretrained_encoder_weights=None,
                                          spm_model_args=spm_args,
                                          fixed_seq_len=args.get('fixed_seq_len'),
                                          num_classes=len(label_map),
                                          with_batch_normalization=args.get('with_batch_normalization') or False)
    model.load_weights(args['model_weights_cp']).expect_partial()
    model.summary()
    readline.parse_and_bind('set editing-mode vi')
    while True:
        sent = input("Paste a document to classify: ")
        subword_ids = spm_encoder(tf.constant([sent]))
        y_probs = model.predict(subword_ids)[0]
        ret = tf.argmax(y_probs).numpy().tolist()
        y_probs_with_labels = list(zip(label_map.values(), y_probs.tolist()))
        print(y_probs_with_labels)
        print(f"Classification result: P({label_map[ret]}) = {y_probs[ret]}")


def evaluate(args):
    x_test, y_test, label_map, test_df = read_tsv_and_numericalize(tsv_file=args['test_tsv'], args=args,
                                                                   also_return_df=True)
    spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': True, 'add_eos': True,
                'lumped_sents_separator': '[SEP]'}
    model, _ = ulmfit_document_classifier(model_type=args['model_type'],
                                          pretrained_encoder_weights=None,
                                          spm_model_args=spm_args,
                                          fixed_seq_len=args.get('fixed_seq_len'),
                                          num_classes=len(label_map),
                                          with_batch_normalization=args.get('with_batch_normalization') or False)
    model.load_weights(args['model_weights_cp']).expect_partial()
    model.summary()
    y_probs_all = model.predict(x_test, batch_size=args['batch_size'],
                                callbacks=[PredictionProgressCallback(x_test.shape[0] // args['batch_size'])])
    y_preds = tf.argmax(y_probs_all, axis=1).numpy()
    y_probs = np.take_along_axis(y_probs_all, y_preds[:, None], axis=1).squeeze(axis=1)
    y_preds = y_preds.tolist()
    y_preds_labels = [label_map[l] for l in y_preds]

    print("\u001b[34m" + classification_report(y_test, y_preds, target_names=list(label_map.values())) + "\u001b[0m")
    if args.get('out_path') is not None:
        df2 = pd.DataFrame.from_dict({'nltext': test_df[args['data_column_name']].tolist(),
                                      'y_probs_all': y_probs_all.tolist(),
                                      'y_probs': y_probs.tolist(),
                                      'gold': [label_map[l] for l in test_df[args['gold_column_name']].tolist()],
                                      'y_preds': y_preds_labels})
        df2['result'] = np.where(df2['gold'] == df2['y_preds'], 'SUCCESS', 'FAIL')
        df2.to_csv(args['out_path'], sep='\t', index=False)


def main(args):
    check_unbounded_training(args.get('fixed_seq_len'), args.get('max_seq_len'))
    x_train, y_train, label_map = read_tsv_and_numericalize(tsv_file=args['train_tsv'], args=args)
    if args.get('test_tsv') is not None:
        x_test, y_test, _, test_df = read_tsv_and_numericalize(tsv_file=args['test_tsv'], args=args,
                                                               also_return_df=True)
    else:
        x_test = y_test = None
    validation_data = (x_test, y_test) if x_test is not None else None
    spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': True, 'add_eos': True,
                'lumped_sents_separator': '[SEP]'}
    model, hub_object = ulmfit_document_classifier(model_type=args['model_type'],
                                                   pretrained_encoder_weights=args['model_weights_cp'],
                                                   spm_model_args=spm_args,
                                                   fixed_seq_len=args.get('fixed_seq_len'),
                                                   num_classes=len(label_map),
                                                   with_batch_normalization=args.get('with_batch_normalization') or False)
    num_steps = (x_train.shape[0] // args['batch_size']) * args['num_epochs']
    print_training_info(args=args, x_train=x_train, y_train=y_train)
    if args.get('lr_finder') is None and args.get('lr_scheduler') == 'stlr':
        scheduler = STLRSchedule(args['lr'], num_steps)
    else:
        scheduler = args['lr']
    optimizer_fn = tf.keras.optimizers.Adam(learning_rate=scheduler)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    callbacks = prepare_keras_callbacks(args=args, model=model, hub_object=hub_object,
                                        monitor_metric='val_sparse_categorical_accuracy' if validation_data is not None
                                        else 'sparse_categorical_accuracy')
    if args.get('lr_scheduler') == '1cycle':
        print("Fitting with one-cycle")
        callbacks.append(OneCycleScheduler(steps=num_steps, lr_max=args['lr']))
    model.compile(optimizer=optimizer_fn,
                  loss=loss_fn,
                  metrics=['sparse_categorical_accuracy'])
    model.summary()
    model.fit(x=x_train, y=y_train, validation_data=validation_data,
              batch_size=args['batch_size'], epochs=args['num_epochs'],
              callbacks=callbacks)
    if args.get('lr_finder'):
        lrfinder = [c for c in callbacks if isinstance(c, LRFinder)][0]
        lrfinder.plot()
    else:
        save_dir = os.path.join(args['out_path'], 'final')
        os.makedirs(save_dir, exist_ok=True)
        model.save_weights(os.path.join(save_dir, 'classifier_final'))


if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument("--train-tsv", required=False, help="Training input file (tsv format)")
    argz.add_argument("--test-tsv", required=False, help="Training test file (tsv format)")
    argz.add_argument('--data-column-name', default='sentence', help="Name of the column containing X data")
    argz.add_argument('--gold-column-name', default='target', help="Name of the gold column in the tsv file")
    argz.add_argument("--label-map", required=True, help="Path to a text file containing labels")
    argz.add_argument("--model-weights-cp", required=True, help="Training: path to *weights* (checkpoint) of "
                                                                "the generic model. Evaluation/Interactive: path to "
                                                                "*weights* produced by this script during training")
    argz.add_argument("--model-type", choices=['from_cp', 'from_hub'], default='from_cp',
                      help="Model type: from_cp = from checkpoint, from_hub = from TensorFlow hub")
    argz.add_argument('--spm-model-file', required=True, help="Path to SentencePiece model file")
    argz.add_argument('--awd-off', required=False, action='store_true', help="Switch off AWD in the training loop.")
    argz.add_argument('--fixed-seq-len', required=False, type=int, help="Fixed sequence length. If unset, the training "
                      "script will use ragged tensors. Otherwise, it will use padding.")
    argz.add_argument('--max-seq-len', required=False, type=int, help="Maximum sequence length")
    argz.add_argument("--batch-size", default=32, type=int, help="Batch size")
    argz.add_argument("--num-epochs", default=1, type=int, help="Number of epochs")
    argz.add_argument("--lr-scheduler", choices=['stlr', '1cycle'], default='stlr', help="Learning rate"
                      "scheduler (slanted triangular or one-cycle)")
    argz.add_argument("--lr", default=0.001, type=float, help="Peak learning rate")
    argz.add_argument("--lr-finder", type=int, help="Run a LR finder for this number of steps")
    argz.add_argument("--interactive", action='store_true', help="Run the script in interactive mode")
    argz.add_argument("--with-batch-normalization", action='store_true', required=False,
                      help="Transform the Y values to be between 0 and max-1.")
    argz.add_argument("--out-path", required=False, help="Training: Path where the trained model (and best checkpoints)"
                                                         " will be saved. Evaluation: path to a TSV file with results")
    argz.add_argument('--save-best', action='store_true', help="Save best checkpoint")
    argz.add_argument('--tensorboard', action='store_true', help="Save Tensorboard logs")
    argz = vars(argz.parse_args())
    if all([argz.get('max_seq_len') and argz.get('fixed_seq_len')]):
        print("You can use either `max_seq_len` with RaggedTensors to restrict the maximum sequence length, or "
              "`fixed_seq_len` with dense tensors to set a fixed sequence length with automatic padding, not both.")
        exit(1)
    if argz.get('interactive') is True:
        interactive_demo(argz)
    elif argz.get('train_tsv'):
        if argz.get('out_path') is None:
            raise ValueError("Please provide an output path where you will store the trained model")
        main(argz)
    elif argz.get('test_tsv'):
        evaluate(argz)
    else:
        print("Unknown action")
        exit(-1)
