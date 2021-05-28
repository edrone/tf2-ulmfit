import os, sys, argparse, readline, math
import json
import nltk
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from modelling_scripts.ulmfit_tf2_heads import ulmfit_document_classifier
from modelling_scripts.ulmfit_tf2 import RaggedSparseCategoricalCrossEntropy, apply_awd_eagerly, AWDCallback
from lm_tokenizers import LMTokenizerFactory
from ulmfit_commons import read_labels, read_numericalize

class STLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, num_steps, cut_frac=0.1, ratio=32):
        self.lr_max = lr_max   # 0.01
        self.T = num_steps     # 900, which is 90 steps over 10 epochs
        self.cut_frac = cut_frac # 0.1
        self.cut = math.floor(num_steps * cut_frac) # 90
        self.ratio = ratio

    def __call__(self, step):
        def warmup(): return step / self.cut
        def cooldown(): return 1 - ((step - self.cut)/(self.cut*(1/(self.cut_frac)-1)))
        def pazz():
            return None
        
        p = tf.cond(tf.less(step, self.cut), warmup, cooldown)
        current_lr = self.lr_max * ( (1 + (p*(self.ratio - 1))) / self.ratio)

        # def printstep():
        #     tf.print(f"Step {step} LR = {current_lr}")
        #     return None 
        # tf.cond(step % 10 == 0, printstep , pazz)
        return current_lr

def interactive_demo(args, label_map):
    raise NotImplementedError

def train_step(*, model, hub_object, loss_fn, optimizer, awd_off=None, x, y, step_info): # todo: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    if awd_off is not True:
        if hub_object is not None: hub_object.apply_awd(0.5)
        else: apply_awd_eagerly(model, 0.5)
    with tf.GradientTape() as tape:
        y_preds = model(x, training=True)
        loss_value = loss_fn(y_true=y, y_pred=y_preds)
        print(f"Step {step_info[0]}/{step_info[1]} | batch loss before applying gradients: {loss_value}")

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def check_unbounded_training(fixed_seq_len, max_seq_len):
    if not any([fixed_seq_len, max_seq_len]):
        print("Warning: you have requested training with an unspecified sequence length. " \
             "This script will not truncate any sequence, but you should make sure that " \
             "all your training examples are reasonably long. You should be fine if your " \
             "training set is split into sentences, but DO make sure that none of them " \
             "runs into thousands of tokens or you will get out-of-memory errors.\n\n")
        sure = "?"
        while sure not in ['y', 'Y', 'n', 'N']:
            sure = input("Are you sure you want to continue? (y/n) ")
        if sure in ['n', 'N']:
            sys.exit(1)

def read_tsv_and_numericalize(*, tsv_file, args, also_return_df=False):
    label_map = read_labels(args['label_map'])
    x_data, y_data, df = read_numericalize(input_file=args['train_tsv'],
                                           spm_model_file=args['spm_model_file'],
                                           label_map=label_map,
                                           max_seq_len = args.get('max_seq_len'),
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
    x_train, y_train, label_map = read_tsv_and_numericalize(tsv_file=args['train_tsv'], args=args)
    if args.get('test_tsv') is not None:
        x_test, y_test, _, test_df = read_tsv_and_numericalize(tsv_file=['test_tsv'], args=args,
                                                               also_return_df=True)
    else:
        x_test = y_test = None
    spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': True, 'add_eos': True,
                'lumped_sents_separator': '[SEP]'}
    ulmfit_classifier, hub_object = ulmfit_document_classifier(model_type=args['model_type'],
                                                               pretrained_encoder_weights=args['model_weights_cp'],
                                                               spm_model_args=spm_args,
                                                               fixed_seq_len=args.get('fixed_seq_len'),
                                                               num_classes=len(label_map))
    ulmfit_classifier.summary()
    num_steps = (x_train.shape[0] // args['batch_size']) * args['num_epochs']
    print(f"************************ TRAINING INFO ***************************\n" \
          f"Shapes - sequence inputs: {x_train.shape}, labels: {y_train.shape}\n" \
          f"Batch size: {args['batch_size']}, Epochs: {args['num_epochs']}, \n" \
          f"Steps per epoch: {x_train.shape[0] // args['batch_size']} \n" \
          f"Total steps: {num_steps}\n" \
          f"******************************************************************")
    scheduler = STLRSchedule(args['lr'], num_steps)
    optimizer_fn = tf.keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.7, beta_2=0.99)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    ulmfit_classifier.compile(optimizer=optimizer_fn,
                              loss=loss_fn,
                              metrics=['sparse_categorical_accuracy'])
    cp_dir = os.path.join(args['out_cp_path'], 'checkpoint')
    final_dir = os.path.join(args['out_cp_path'], 'final')
    for d in [cp_dir, final_dir]: os.makedirs(d, exist_ok=True)
    callbacks = []
    if not args.get('awd_off'):
        callbacks.append(AWDCallback(model_object=ulmfit_classifier if hub_object is None else None,
                                     hub_object=hub_object))
    if args.get('tensorboard'):
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='tboard_logs', update_freq='batch'))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(cp_dir, 'classifier_model'),
                                           monitor='val_sparse_categorical_accuracy',
                                           save_best_only=True,
                                           save_weights_only=True))
    
    validation_data = (x_test, y_test) if x_test is not None else None
    #exit(0)
    ulmfit_classifier.fit(x=x_train, y=y_train, batch_size=args['batch_size'],
                          validation_data=validation_data,
                          epochs=args['num_epochs'],
                          callbacks=callbacks)
    ulmfit_classifier.save_weights(os.path.join(final_dir, 'classifier_final'))
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
    argz.add_argument("--interactive", action='store_true', help="Run the script in interactive mode")
    argz.add_argument("--label-map", required=True, help="Path to a text file containing labels")
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
