"""
Train an ULMFiT regressor model

A regressor model outputs a real-number value whereas a classifier model outputs class probabilities.

If `--normalize-labels` is passed, the gold values are rescaled to a range between 0 and 1
"""
import argparse
import os
import readline

import numpy as np
import pandas as pd
import tensorflow as tf

from lm_tokenizers import LMTokenizerFactory
from ulmfit_commons import read_numericalize, check_unbounded_training, print_training_info, prepare_keras_callbacks
from ulmfit_tf2 import STLRSchedule, OneCycleScheduler, PredictionProgressCallback
from ulmfit_tf2_heads import ulmfit_regressor


def interactive_demo(args):
    spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': True, 'add_eos': True,
                'lumped_sents_separator': '[SEP]'}
    spm_encoder = LMTokenizerFactory.get_tokenizer(tokenizer_type='spm_tf_text',
                                                   tokenizer_file=args['spm_model_file'],
                                                   fixed_seq_len=args.get('fixed_seq_len'),
                                                   add_bos=True, add_eos=True)
    ulmfit_regressor_model, hub_object = ulmfit_regressor(model_type=args['model_type'],
                                                          pretrained_encoder_weights=None,
                                                          spm_model_args=spm_args,
                                                          fixed_seq_len=args.get('fixed_seq_len'),
                                                          with_batch_normalization=args.get('with_batch_normalization') or False)
    ulmfit_regressor_model.load_weights(args['model_weights_cp']).expect_partial()
    ulmfit_regressor_model.summary()
    readline.parse_and_bind('set editing-mode vi')
    while True:
        sent = input("Paste a document to classify using a regressor: ")
        subword_ids = spm_encoder(tf.constant([sent]))
        y_hat = ulmfit_regressor_model.predict(subword_ids)[0]
        print(f"Score: = {y_hat}")

def read_tsv_and_numericalize(*, tsv_file, args, also_return_df=False):
    x_data, y_data, df = read_numericalize(input_file=tsv_file,
                                           spm_model_file=args['spm_model_file'],
                                           max_seq_len = args.get('max_seq_len'),
                                           fixed_seq_len = args.get('fixed_seq_len'),
                                           x_col=args['data_column_name'],
                                           y_col=args['gold_column_name'],
                                           sentence_tokenize=False,
                                           cut_off_final_token=False)
    if args.get('fixed_seq_len') is not None:
        x_data = tf.constant(x_data, dtype=tf.int32)
    else:
        x_data = tf.ragged.constant(x_data, dtype=tf.int32)
    y_data = tf.constant(y_data, dtype=tf.float32) # real-valued numbers
    ret = {'x_data': x_data,
           'y_data': y_data}
    if args.get('normalize_labels') is True:
        max_label_value = tf.reduce_max(y_data)
        y_data_norm = (y_data - 1.0) / (max_label_value - 1.0)
        y_data_unscaled = ret.pop('y_data')
        ret['y_data'] = y_data_norm
        ret['y_data_orig'] = y_data_unscaled
        df['y_data_orig'] = y_data_unscaled
    if also_return_df:
        ret['df'] = df
    return ret

def get_keras_regression_objects(loss_fn_name):
    if loss_fn_name == 'mae':
        return tf.keras.losses.MeanAbsoluteError(), tf.keras.metrics.MeanAbsoluteError()
    elif loss_fn_name == 'mse':
        return tf.keras.losses.MeanSquaredError(), tf.keras.metrics.MeanSquaredError()
    else:
        raise ValueError(f"Unknown loss function name {loss_fn_name}")

def evaluate(args):
    read_data = read_tsv_and_numericalize(tsv_file=args['test_tsv'],
                                          args=args,
                                          also_return_df=True)
    x_test = read_data['x_data']
    y_test = read_data['y_data']
    y_test_orig = y_test if read_data.get('y_data_orig') is None else read_data['y_data_orig']
    test_df = read_data['df']
    spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': True, 'add_eos': True,
                'lumped_sents_separator': '[SEP]'}
    ulmfit_regressor_model, hub_object = ulmfit_regressor(model_type=args['model_type'],
                                                          pretrained_encoder_weights=None,
                                                          spm_model_args=spm_args,
                                                          fixed_seq_len=args.get('fixed_seq_len'),
                                                          with_batch_normalization=args.get('with_batch_normalization') or False)
    ulmfit_regressor_model.load_weights(args['model_weights_cp'])
    ulmfit_regressor_model.summary()
    y_preds = ulmfit_regressor_model.predict(x_test, batch_size=args['batch_size'],
                                             callbacks=[PredictionProgressCallback(x_test.shape[0] // args['batch_size'])])
    y_test = y_test.numpy().tolist()
    y_preds = np.squeeze(y_preds)
    if args.get('normalize_labels'):
        max_label_value = tf.reduce_max(y_test_orig)
        y_preds_rescaled =  ((max_label_value - 1.0) * (y_preds)) + 1.0
    else:
        y_preds_rescaled = y_preds
    y_preds = y_preds.tolist(); y_preds_rescaled = y_preds_rescaled.numpy().tolist()
    df2 = pd.DataFrame.from_dict({'nltext': test_df[args['data_column_name']].tolist(),
                                  'gold': y_test,
                                  'gold_unscaled': y_test_orig,
                                  'y_preds': y_preds,
                                  'y_preds_rescaled': y_preds_rescaled})
    if args['loss_fn'] == 'mae':
        df2['error'] = (df2['y_preds'] - df2['gold']).abs()
    elif args['loss_fn'] == 'mse':
        df2['error'] = (df2['y_preds'] - df2['gold'])**2
    print(f"Result metric ({args['loss_fn']}): {df2['error'].mean()}")
    if args.get('out_path') is not None:
        df2.to_csv(args['out_path'], sep='\t', index=None)
 
def main(args):
    check_unbounded_training(args.get('fixed_seq_len'), args.get('max_seq_len'))
    read_data = read_tsv_and_numericalize(tsv_file=args['train_tsv'], args=args)
    x_train = read_data['x_data']
    y_train = read_data['y_data']
    print(y_train)
    if args.get('test_tsv') is not None:
        read_data = read_tsv_and_numericalize(tsv_file=args['test_tsv'], args=args, also_return_df=True)
        x_test = read_data['x_data']
        y_test = read_data['y_data']
        test_df = read_data['df']
        print(y_test)
    else:
        x_test = y_test = None
    validation_data = (x_test, y_test) if x_test is not None else None
    spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': True, 'add_eos': True,
                'lumped_sents_separator': '[SEP]'}
    ulmfit_regressor_model, hub_object = ulmfit_regressor(model_type=args['model_type'],
                                                    pretrained_encoder_weights=args['model_weights_cp'],
                                                    spm_model_args=spm_args,
                                                    fixed_seq_len=args.get('fixed_seq_len'),
                                                    with_batch_normalization=args.get('with_batch_normalization') or False)
    ulmfit_regressor_model.summary()
    num_steps = (x_train.shape[0] // args['batch_size']) * args['num_epochs']
    print_training_info(args=args, x_train=x_train, y_train=y_train)
    if args.get('lr_scheduler') == 'stlr':
        scheduler = STLRSchedule(args['lr'], num_steps)
    else:
        scheduler = args['lr']
    optimizer_fn = tf.keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.7, beta_2=0.99)
    loss_fn, loss_metric = get_keras_regression_objects(args['loss_fn'])
    monitor_metric = 'mean_absolute_error' if args['loss_fn'] == 'mae' else 'mean_squared_error'
    callbacks = prepare_keras_callbacks(args=args, model=ulmfit_regressor_model, hub_object=hub_object,
                                        monitor_metric=f'val_{monitor_metric}' if validation_data is not None \
                                        else monitor_metric)
    if args.get('lr_scheduler') == '1cycle':
        print("Fitting with one-cycle")
        callbacks.append(OneCycleScheduler(steps=num_steps, lr_max=args['lr']))
    ulmfit_regressor_model.compile(optimizer=optimizer_fn,
                                   loss=loss_fn,
                                   metrics=[loss_metric])
    
    ulmfit_regressor_model.fit(x=x_train, y=y_train, validation_data=validation_data,
                               batch_size=args['batch_size'],
                               epochs=args['num_epochs'],
                               callbacks=callbacks)

    save_dir = os.path.join(args['out_path'], 'final')
    os.makedirs(save_dir, exist_ok=True)
    ulmfit_regressor_model.save_weights(os.path.join(save_dir, 'regressor_final'))

if __name__ == "__main__":
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
    argz.add_argument('--awd-off', required=False, action='store_true', help="Switch off AWD in the training loop.")
    argz.add_argument('--fixed-seq-len', required=False, type=int, help="Fixed sequence length. If unset, the training "\
                                                                        "script will use ragged tensors. Otherwise, it will use padding.")
    argz.add_argument('--max-seq-len', required=False, type=int, help="Maximum sequence length. Only makes sense with RaggedTensors.")
    argz.add_argument("--batch-size", default=32, type=int, help="Batch size")
    argz.add_argument("--num-epochs", default=1, type=int, help="Number of epochs")
    argz.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    argz.add_argument("--lr-scheduler", choices=['stlr', '1cycle'], default='stlr', help="Learning rate"
                      "scheduler (slanted triangular or one-cycle)")
    argz.add_argument("--loss-fn", default='mae', choices=['mae', 'mse'], help="Loss function for regression (MAE or MSE).")
    argz.add_argument("--with-batch-normalization", action='store_true', required=False, help="Use batch normalization (looks broken)")
    argz.add_argument("--normalize-labels", action='store_true', required=False, help="Transform the Y values to be between 0 and max-1.")
    argz.add_argument("--interactive", action='store_true', help="Run the script in interactive mode")
    argz.add_argument("--out-path", required=False, help="At training: directory to save the checkpoints and the final model. " \
                                                         "At evaluation: path where the TSV file with results will be saved.")
    argz.add_argument('--tensorboard', action='store_true', help="Save Tensorboard logs")
    argz.add_argument('--save-best', action='store_true', help="Save best checkpoint")
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
