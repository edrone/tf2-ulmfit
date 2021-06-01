import os, sys, argparse, readline
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from modelling_scripts.ulmfit_tf2_heads import ulmfit_last_hidden_state
from modelling_scripts.ulmfit_tf2 import RaggedSparseCategoricalCrossEntropy, STLRSchedule, AWDCallback
from ulmfit_text_classifier import read_tsv_and_numericalize
from ulmfit_commons import check_unbounded_training
from lm_tokenizers import LMTokenizerFactory

def interactive_demo(args, label_map):
    raise NotImplementedError
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
    ulmfit_classifier_lasthidden, hub_object = ulmfit_last_hidden_state(model_type=args['model_type'],
                                                               pretrained_encoder_weights=args['model_weights_cp'],
                                                               spm_model_args=spm_args,
                                                               fixed_seq_len=args.get('fixed_seq_len'))
    drop1 = tf.keras.layers.Dropout(0.4)(ulmfit_classifier_lasthidden.output)
    fc1 = tf.keras.layers.Dense(50, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(0.1)(fc1)
    fc_final = tf.keras.layers.Dense(len(label_map), activation='softmax')(drop2)
    plain_document_classifier_model = tf.keras.models.Model(inputs=ulmfit_classifier_lasthidden.input,
                                                      outputs=fc_final)
    plain_document_classifier_model.summary()
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
    plain_document_classifier_model.compile(optimizer=optimizer_fn,
                                            loss=loss_fn,
                                            metrics=['sparse_categorical_accuracy'])
    cp_dir = os.path.join(args['out_cp_path'], 'checkpoint')
    final_dir = os.path.join(args['out_cp_path'], 'final')
    for d in [cp_dir, final_dir]: os.makedirs(d, exist_ok=True)
    callbacks = []
    if not args.get('awd_off'):
        callbacks.append(AWDCallback(model_object=plain_document_classifier_model if hub_object is None else None,
                                     hub_object=hub_object))
    if args.get('tensorboard'):
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='tboard_logs', update_freq='batch'))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(cp_dir, 'classifier_model'),
                                           monitor='val_sparse_categorical_accuracy',
                                           save_best_only=True,
                                           save_weights_only=True))
    
    validation_data = (x_test, y_test) if x_test is not None else None
    #exit(0)
    plain_document_classifier_model.fit(x=x_train, y=y_train, batch_size=args['batch_size'],
                                        validation_data=validation_data,
                                        epochs=args['num_epochs'],
                                        callbacks=callbacks)
    plain_document_classifier_model.save_weights(os.path.join(final_dir, 'plain_classifier_final'))

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
    argz.add_argument("--label-map", required=True, help="Path to a text file containing labels.")
    argz.add_argument("--out-cp-path", required=False, help="(Training only): Checkpoint name to save every 10 steps")
    argz = vars(argz.parse_args())
    if all([argz.get('max_seq_len') and argz.get('fixed_seq_len')]):
        print("You can use either `max_seq_len` with RaggedTensors to restrict the maximum sequence length, or `fixed_seq_len` with dense "\
              "tensors to set a fixed sequence length with automatic padding, not both.")
        exit(1)
    if argz.get('interactive') is True:
        interactive_demo(argz)
    elif argz.get('train_tsv'):
        if argz.get('out_cp_path') is None:
            raise ValueError("Please provide an output path where you will store the trained model")
        main(argz)
    elif argz.get('test_tsv'):
        evaluate(argz)
    else:
        print("Unknown action")
        main(argz)
