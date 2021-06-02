import os, sys, argparse, readline, math
import json
import nltk
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from modelling_scripts.ulmfit_tf2_heads import ulmfit_regressor
from modelling_scripts.ulmfit_tf2 import apply_awd_eagerly, AWDCallback, STLRSchedule
from lm_tokenizers import LMTokenizerFactory
from ulmfit_commons import read_numericalize, check_unbounded_training

def interactive_demo(args):
    raise NotImplementedError

def read_tsv_and_numericalize(*, tsv_file, args, also_return_df=False):
    x_data, y_data, df = read_numericalize(input_file=args['train_tsv'],
                                           spm_model_file=args['spm_model_file'],
                                           max_seq_len = args.get('max_seq_len'),
                                           x_col=args['data_column_name'],
                                           y_col=args['gold_column_name'],
                                           sentence_tokenize=True,
                                           cut_off_final_token=False)
    if args.get('fixed_seq_len') is not None:
        x_data = tf.constant(x_data, dtype=tf.int32)
    else:
        x_data = tf.ragged.constant(x_data, dtype=tf.int32)
    y_data = tf.constant(y_data, dtype=tf.float32) # real-valued numbers
    if args.get('normalize_labels') is True:
        max_label_value = tf.reduce_max(y_data)
        y_data = (y_data - 1.0) / (max_label_value - 1.0)
    if also_return_df is True:
        return x_data, y_data, df
    else:
        return x_data, y_data

def get_keras_regression_objects(loss_fn_name):
    if loss_fn_name == 'mae':
        return tf.keras.losses.MeanAbsoluteError(), tf.keras.metrics.MeanAbsoluteError()
    elif loss_fn_name == 'mse':
        return tf.keras.losses.MeanSquaredError(), tf.keras.metrics.MeanSquaredError()
    else:
        raise ValueError(f"Unknown loss function name {loss_fn_name}")

def evaluate(args):
    x_data, labels, spm_args = read_numericalize(input_file=args['test_tsv'],
                                                 spm_model_file=args['spm_model_file'],
                                                 label_map=label_map,
                                                 max_seq_len = args.get('max_seq_len'),
                                                 x_col=args['data_column_name'],
                                                 y_col=args['gold_column_name'])
    if args.get('fixed_seq_len') is not None:
        raise NotImplementedError("Not implemented yet")
    else:
        ulmfit_classifier, _ = ulmfit_document_classifier(model_type=args['model_type'],
                                                          pretrained_encoder_weights=args['model_weights_cp'],
                                                          spm_model_args=spm_args,
                                                          fixed_seq_len=args.get('fixed_seq_len'),
                                                          num_classes=args['num_classes'])
    ulmfit_classifier.summary()
    print(f"Shapes - sequence inputs: {x_data.shape}, labels: {labels.shape}")
    return ulmfit_classifier, x_data, labels
    
def main(args):
    check_unbounded_training(args.get('fixed_seq_len'), args.get('max_seq_len'))
    x_train, y_train = read_tsv_and_numericalize(tsv_file=args['train_tsv'], args=args)
    print(y_train)
    if args.get('test_tsv') is not None:
        x_test, y_test, test_df = read_tsv_and_numericalize(tsv_file=['test_tsv'], args=args,
                                                            also_return_df=True)
    else:
        x_test = y_test = None
    spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': True, 'add_eos': True,
                'lumped_sents_separator': '[SEP]'}
    ulmfit_regressor_model, hub_object = ulmfit_regressor(model_type=args['model_type'],
                                                    pretrained_encoder_weights=args['model_weights_cp'],
                                                    spm_model_args=spm_args,
                                                    fixed_seq_len=args.get('fixed_seq_len'),
                                                    with_batch_normalization=False)
    ulmfit_regressor_model.summary()
    num_steps = (x_train.shape[0] // args['batch_size']) * args['num_epochs']
    print(f"************************ TRAINING INFO ***************************\n" \
          f"Shapes - sequence inputs: {x_train.shape}, labels: {y_train.shape}\n" \
          f"Batch size: {args['batch_size']}, Epochs: {args['num_epochs']}, \n" \
          f"Steps per epoch: {x_train.shape[0] // args['batch_size']} \n" \
          f"Total steps: {num_steps}\n" \
          f"******************************************************************")
    scheduler = STLRSchedule(args['lr'], num_steps)
    optimizer_fn = tf.keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.7, beta_2=0.99)
    loss_fn, loss_metric = get_keras_regression_objects(args['loss_fn'])
    ulmfit_regressor_model.compile(optimizer=optimizer_fn,
                                   loss=loss_fn,
                                   metrics=[loss_metric])
    cp_dir = os.path.join(args['out_cp_path'], 'checkpoint')
    final_dir = os.path.join(args['out_cp_path'], 'final')
    for d in [cp_dir, final_dir]: os.makedirs(d, exist_ok=True)
    callbacks = []
    if not args.get('awd_off'):
        callbacks.append(AWDCallback(model_object=ulmfit_regressor_model if hub_object is None else None,
                                     hub_object=hub_object))
    if args.get('tensorboard'):
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='tboard_logs', update_freq='batch'))
    # fixme: validation metric below
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(cp_dir, 'regressor_model'),
                                           monitor='val_mean_absolute_error',
                                           save_best_only=True,
                                           save_weights_only=True))
    
    validation_data = (x_test, y_test) if x_test is not None else None
    ulmfit_regressor_model.fit(x=x_train, y=y_train, batch_size=args['batch_size'],
                               validation_data=validation_data,
                               epochs=args['num_epochs'],
                               callbacks=callbacks)
    ulmfit_regressor_model.save_weights(os.path.join(final_dir, 'regressor_final'))
    return ulmfit_classifier, x_train, y_train, loss_fn, optimizer

if __name__ == "__main__":
    # TODO: the weights checkpoint quirk should be done away with, but to serialize anything custom into a SavedModel
    # especially if that thing contains RaggedTensors is kind of nightmarish...
    argz = argparse.ArgumentParser()
    argz.add_argument("--train-tsv", required=False, help="Training input file (tsv format)")
    argz.add_argument("--test-tsv", required=False, help="Training test file (tsv format)")
    argz.add_argument('--data-column-name', default='sentence', help="Name of the column containing X data")
    argz.add_argument('--gold-column-name', default='target', help="Name of the gold column in the tsv file")
    argz.add_argument("--model-weights-cp", required=True, help="For training: path to *weights* (checkpoint) of " \
                                                                "the generic model." \
                                                                "For demo: path to *weights* produced by this script")
    argz.add_argument("--model-type", choices=['from_cp', 'from_hub'], default='from_cp', \
                                                           help="Model type: from_cp = from checkpoint, from_hub = from TensorFlow hub")
    argz.add_argument('--spm-model-file', required=True, help="Path to SentencePiece model file")
    argz.add_argument('--awd-off', required=False, action='store_true', help="Switch off AWD in the training loop.") # todo: set AWD rate
    argz.add_argument('--fixed-seq-len', required=False, type=int, help="Fixed sequence length. If unset, the training "\
                                                                        "script will use ragged tensors. Otherwise, it will use padding.")
    argz.add_argument('--max-seq-len', required=False, type=int, help="Maximum sequence length. Only makes sense with RaggedTensors.")
    argz.add_argument("--batch-size", default=32, type=int, help="Batch size")
    argz.add_argument("--num-epochs", default=1, type=int, help="Number of epochs")
    argz.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    argz.add_argument("--loss-fn", default='mae', choices=['mae', 'mse'], help="Loss function for regression (MAE or MSE).")
    argz.add_argument("--normalize-labels", action='store_true', required=False, help="Transform the Y values to be between 0 and max-1.")
    argz.add_argument("--interactive", action='store_true', help="Run the script in interactive mode")
    argz.add_argument("--out-cp-path", default="out", help="(Training only): Directory to save the checkpoints and the final model")
    argz.add_argument('--tensorboard', action='store_true', help="Save Tensorboard logs")
    argz = vars(argz.parse_args())
    if all([argz.get('max_seq_len') and argz.get('fixed_seq_len')]):
        print("You can use either `max_seq_len` with RaggedTensors to restrict the maximum sequence length, or `fixed_seq_len` with dense "\
              "tensors to set a fixed sequence length with automatic padding, not both.")
        exit(1)
    if argz.get('interactive') is True:        
        interactive_demo(argz)
    if argz.get('train_tsv'):
        main(argz)
    elif argz.get('test_tsv'):
        evaluate(argz)
    else:
        print("Unknown action")
        exit(-1)
