import os, sys, argparse, readline
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from modelling_scripts.ulmfit_tf2_heads import ulmfit_last_hidden_state
from modelling_scripts.ulmfit_tf2 import RaggedSparseCategoricalCrossEntropy, STLRSchedule
from ulmfit_tf_text_classifier import read_tsv_and_numericalize
from ulmfit_commons import check_unbounded_training, prepare_keras_callbacks, print_training_info, read_labels
from lm_tokenizers import LMTokenizerFactory

def interactive_demo(args):
    raise NotImplementedError
    spm_encoder = LMTokenizerFactory.get_tokenizer(tokenizer_type='spm_tf_text', \
                                               tokenizer_file=args['spm_model_file'], \
                                               add_bos=True, add_eos=True)
    label_map = read_labels(args['label_map'])
    model, _ = build_lasthidden_classifier_model(args=args, num_labels=len(label_map),
                                                 restore_encoder=False)
    model.summary()
    model.load_weights(args['model_weights_cp'])
    print("Done")
    readline.parse_and_bind('set editing-mode vi')
    while True:
        sent = input("Write a document to classify: ")
        subword_ids = spm_encoder(tf.constant([sent]))
        # subwords = spmproc.id_to_string(subword_ids)[0].numpy().tolist() # this contains bytes, not strings
        # subwords = [s.decode() for s in subwords]
        ret = tf.argmax(model.predict(subword_ids)[0]).numpy().tolist()
        print(ret)

def build_lasthidden_classifier_model(*, args, num_labels, restore_encoder=False):
    """
    Build a simple document classifier.

    The ULMFiT paper uses a concatenated vector of the last hidden state,
    max pooling and average pooling for document classification. Here
    we only use the last hidden state.

    :param dict args:       Arguments dictionary (see the argparse fields)
    :param int num_labels:  Number of labels (target classes)
    :param bool restore_encoder: Whether or not the RNN encoder weights should be
                                 restored from args['model_weights_cp'] path
    :return: a Keras functional model with numericalized inputs and softmaxed outputs
    """
    spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': True, 'add_eos': True,
                'lumped_sents_separator': '[SEP]'}
    weights_path = None if restore_encoder is False else args['model_weights_cp']
    ulmfit_lasthidden, hub_object = ulmfit_last_hidden_state(model_type=args['model_type'],
                                                             pretrained_encoder_weights=weights_path,
                                                             spm_model_args=spm_args,
                                                             fixed_seq_len=args.get('fixed_seq_len'))
    drop1 = tf.keras.layers.Dropout(0.4)(ulmfit_lasthidden.output)
    fc1 = tf.keras.layers.Dense(50, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(0.1)(fc1)
    fc_final = tf.keras.layers.Dense(num_labels, activation='softmax')(drop2)
    plain_document_classifier_model = tf.keras.models.Model(inputs=ulmfit_lasthidden.input,
                                                            outputs=fc_final)
    return plain_document_classifier_model, hub_object

def main(args):
    # Step 1. Read data into memory
    check_unbounded_training(args.get('fixed_seq_len'), args.get('max_seq_len'))
    x_train, y_train, label_map = read_tsv_and_numericalize(tsv_file=args['train_tsv'], args=args)
    if args.get('test_tsv') is not None:
        x_test, y_test, _, test_df = read_tsv_and_numericalize(tsv_file=['test_tsv'], args=args,
                                                               also_return_df=True)
    else:
        x_test = y_test = None
    validation_data = (x_test, y_test) if x_test is not None else None

    # Step 2. Build the classifier model, set up the optimizer and callbacks
    model, hub_object = build_lasthidden_classifier_model(args=args, num_labels=len(label_map),
                                                          restore_encoder=True)
    num_steps = (x_train.shape[0] // args['batch_size']) * args['num_epochs']
    print_training_info(args=args, x_train=x_train, y_train=y_train)
    scheduler = STLRSchedule(args['lr'], num_steps)
    optimizer_fn = tf.keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.7, beta_2=0.99)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    callbacks = prepare_keras_callbacks(args=args, model=model, hub_object=hub_object,
                                        monitor_metric = 'val_sparse_categorical_accuracy' \
                                                         if validation_data is not None \
                                                         else 'sparse_categorical_accuracy')
    model.summary()
    model.compile(optimizer=optimizer_fn,
                  loss=loss_fn,
                  metrics=['sparse_categorical_accuracy'])

    # Step 3. Run the training
    model.fit(x=x_train, y=y_train, validation_data=validation_data,
              batch_size=args['batch_size'],
              epochs=args['num_epochs'],
              callbacks=callbacks)

    # Step 4. Save weights
    save_dir = os.path.join(args['out_cp_path'], 'final')
    os.makedirs(save_dir, exist_ok=True)
    model.save_weights(os.path.join(final_dir, 'lasthidden_classifier_model'))

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
