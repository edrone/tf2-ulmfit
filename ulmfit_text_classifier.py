import sys, argparse, readline
import json
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from modelling_scripts.ulmfit_tf2_heads import *
from modelling_scripts.ulmfit_tf2 import RaggedSparseCategoricalCrossEntropy
from ulmfit_commons import apply_awd_eagerly
from lm_tokenizers import LMTokenizerFactory

DEFAULT_LABEL_MAP = {0: '__label__meta_zero', 1: '__label__meta_plus_m', 2: '__label__meta_minus_m', 3:'__label__meta_amb'} # fixme: label map should not be hardcoded (maybe pass as parameter?)

def read_numericalize(*, input_file, spm_model_file, label_map, max_seq_len):
    df = pd.read_csv(args['train_tsv'], sep='\t')
    df['target'].replace({v:k for k,v in label_map.items()}, inplace=True)
    df['sentence'] = df['sentence'].str.replace(' . ', '[SEP]', regex=False)
    spm_args = {'spm_path': args['spm_model_file'],
                'add_bos': True,
                'add_eos': False,
                'lumped_sents_separator': '[SEP]'
    }
    spm_layer = SPMNumericalizer(**spm_args)
    spm_args['spm_model_file'] = spm_args.pop('spm_path') # gosh...
    x_data = spm_layer(df['sentence'].tolist())
    if args.get('max_seq_len') is not None:
        x_data = x_data[:, :args['max_seq_len']]
    #spmproc = LMTokenizerFactory.get_tokenizer(tokenizer_type='spm', \
    #                                           tokenizer_file=args['spm_model_file'], \
    #                                           add_bos=False, add_eos=False) # bos and eos will need to be added manually
    labels = df['target'].to_numpy()
    return x_data, labels

def interactive_demo(args, label_map):
    #spmproc = LMTokenizerFactory.get_tokenizer(tokenizer_type='spm', \
    #                                           tokenizer_file=args['spm_model_file'], \
    #                                           add_bos=True, add_eos=True) # bos and eos will need to be added manually
    spm_encoder = LMTokenizerFactory.get_tokenizer(tokenizer_type='spm_tf_text', \
                                               tokenizer_file=args['spm_model_file'], \
                                               add_bos=True, add_eos=True) # bos and eos will need to be added manually
    spm_args = {'spm_model_file': args['spm_model_file'],
                'add_bos': False,
                'add_eos': False,
                'lumped_sents_separator': '[SEP]'
    }
    spmproc = spm_encoder.spmproc
    ulmfit_tagger, hub_object = ulmfit_sequence_tagger(model_type=args['model_type'],
                                                       pretrained_encoder_weights=None,
                                                       spm_model_args=spm_args,
                                                       fixed_seq_len=args.get('fixed_seq_len'),
                                                       num_classes=len(label_map))
    ulmfit_tagger.summary()
    ulmfit_tagger.load_weights(args['model_weights_cp'])
    print("Done")
    ulmfit_tagger.summary()
    readline.parse_and_bind('set editing-mode vi')
    while True:
        sent = input("Write a sentence to tag: ")
        # Our SPMNumericalizer already outputs a RaggedTensor, but in the line below we access
        # the underlying object directly on purpose, so we have to convert it from regular to ragged tensor ourselves.
        subword_ids = spmproc.tokenize(sent)
        subword_ids = tf.RaggedTensor.from_tensor(tf.expand_dims(subword_ids, axis=0))
        subwords = spmproc.id_to_string(subword_ids)[0].numpy().tolist() # this contains bytes, not strings
        subwords = [s.decode() for s in subwords]
        ret = tf.argmax(ulmfit_tagger.predict(subword_ids)[0], axis=1).numpy().tolist()
        for subword, category in zip(subwords, ret):
            print("{:<15s}{:>4s}".format(subword, label_map[category]))

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

def evaluate(args, label_map):
    x_data, labels = read_numericalize(input_file=args['test_tsv'],
                                       spm_model_file=args['spm_model_file'],
                                       label_map=label_map,
                                       max_seq_len = args.get('max_seq_len'))
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
    
    
    
def main(args, label_map):
    check_unbounded_training(args.get('fixed_seq_len'), args.get('max_seq_len'))
    x_data, labels = read_numericalize(input_file=args['train_tsv'],
                                       spm_model_file=args['spm_model_file'],
                                       label_map=label_map,
                                       max_seq_len = args.get('max_seq_len'))
    if args.get('fixed_seq_len') is not None:
        raise NotImplementedError("Not implemented yet")
    else:
        ulmfit_classifier, hub_object = ulmfit_document_classifier(model_type=args['model_type'],
                                                                   pretrained_encoder_weights=args['model_weights_cp'],
                                                                   spm_model_args=spm_args,
                                                                   fixed_seq_len=args.get('fixed_seq_len'),
                                                                   num_classes=args['num_classes'])
    ulmfit_classifier.summary()
    print(f"Shapes - sequence inputs: {x_data.shape}, labels: {labels.shape}")
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    ##### For RaggedTensors and variable-length sequences we have to use the GradientTape ########
    ulmfit_classifier.compile(optimizer='adam', loss=loss_fn, metrics=['sparse_categorical_accuracy'])
    batch_size = args['batch_size']
    steps_per_epoch = x_data.shape[0] // batch_size
    for epoch in range(args['num_epochs']):
        for step in range(steps_per_epoch - 1):
            if step % 25 == 0:
                print("Saving weights...")
                ulmfit_classifier.save_weights(args['out_cp_name'])
            train_step(model=ulmfit_classifier,
                       hub_object=hub_object,
                       loss_fn=loss_fn, optimizer=optimizer,
                       x=x_data[(step*batch_size):(step+1)*batch_size],
                       y=labels[(step*batch_size):(step+1)*batch_size],
                       step_info=(step, steps_per_epoch))
        # TODO: add shuffling after every epoch
    return ulmfit_classifier, x_data, labels, loss_fn, optimizer

if __name__ == "__main__":
    # TODO: the weights checkpoint quirk should be done away with, but to serialize anything custom into a SavedModel
    # especially if that thing contains RaggedTensors is kind of nightmarish...
    argz = argparse.ArgumentParser()
    argz.add_argument("--train-tsv", required=False, help="Training input file (tsv format)")
    argz.add_argument("--test-tsv", required=False, help="Training test file (tsv format)")
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
    argz.add_argument("--interactive", action='store_true', help="Run the script in interactive mode")
    argz.add_argument("--label-map", required=False, help="Path to a JSON file containing labels. If not given, " \
                                                          "4 classes will be used.")
    argz.add_argument("--num-classes", type=int, default=4, help="Number of label categories")
    argz.add_argument("--out-cp-name", default="polemo_results", help="(Training only): Checkpoint name to save every 10 steps")
    argz = vars(argz.parse_args())
    if all([argz.get('max_seq_len') and argz.get('fixed_seq_len')]):
        print("You can use either `max_seq_len` with RaggedTensors to restrict the maximum sequence length, or `fixed_seq_len` with dense "\
              "tensors to set a fixed sequence length with automatic padding, not both.")
        exit(1)
    if argz.get('label_map') is not None:
        label_map = open(argz['label_map'], 'r', encoding='utf-8').readlines()
        label_map = {k:v.strip() for k,v in enumerate(label_map) if len(v)>0}
    else:
        label_map = DEFAULT_LABEL_MAP
    if argz.get('interactive') is True:        
        interactive_demo(argz, label_map)
    elif argz.get('test_tsv'):
        evaluate(argz, label_map)
    else:
        main(argz, label_map)
