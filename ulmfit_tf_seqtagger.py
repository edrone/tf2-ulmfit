import sys, argparse, readline
import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from ptools.lipytools.little_methods import r_jsonl
from modelling_scripts.ulmfit_tf2_heads import *
from modelling_scripts.ulmfit_tf2 import RaggedSparseCategoricalCrossEntropy, apply_awd_eagerly
from lm_tokenizers import LMTokenizerFactory

DEFAULT_LABEL_MAP = {0: 'O', 1: 'B-N', 2: 'I-N'} # fixme: label map should not be hardcoded (maybe pass as parameter?)

def tokenize_and_align_labels(spmproc, ddpl_iob, max_seq_len):
    """
    Performs Sentencepiece tokenization on an already whitespace-tokenized text
    and aligns labels to subwords
    """

    print(f"Tokenizing and aligning {len(ddpl_iob)} examples...")
    if max_seq_len is not None:
        print(f"Note: inputs will be truncated to the first {max_seq_len - 2} tokens")
    tokenized = []
    numericalized = []
    labels = []
    for sent in ddpl_iob:
        sentence_tokens = []
        sentence_ids = []
        sentence_labels = []
        for whitespace_token in sent:
            subwords = spmproc.encode_as_pieces(whitespace_token[0])
            sentence_tokens.extend(subwords)
            sentence_ids.extend(spmproc.encode_as_ids(whitespace_token[0]))
            sentence_labels.extend([whitespace_token[1]]*len(subwords))
        if max_seq_len is not None:
            sentence_tokens = sentence_tokens[:max_seq_len-2] # minus 2 tokens for BOS and EOS since the encoder was trained on sentences with these markers
            sentence_ids = sentence_ids[:max_seq_len-2]
            sentence_labels = sentence_labels[:max_seq_len-2]
        sentence_tokens = [spmproc.id_to_piece(spmproc.bos_id())] + \
                          sentence_tokens + \
                          [spmproc.id_to_piece(spmproc.eos_id())]
        sentence_ids = [spmproc.bos_id()] + sentence_ids + [spmproc.eos_id()]
        sentence_labels = [0] + sentence_labels + [0]
        tokenized.append(sentence_tokens)
        numericalized.append(sentence_ids)
        labels.append(sentence_labels)
    return tokenized, numericalized, labels

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

def main(args):
    check_unbounded_training(args.get('fixed_seq_len'), args.get('max_seq_len'))
    ddpl_iob = r_jsonl(args['ddpl_iob'])
    spm_args = {'spm_model_file': args['spm_model_file'],
                'add_bos': False,
                'add_eos': False,
                'lumped_sents_separator': '[SEP]'
    }
    spmproc = LMTokenizerFactory.get_tokenizer(tokenizer_type='spm', \
                                               tokenizer_file=args['spm_model_file'], \
                                               add_bos=False, add_eos=False) # bos and eos will need to be added manually
    tokenized, numericalized, labels = tokenize_and_align_labels(spmproc, ddpl_iob, args.get('max_seq_len'))
    print(f"Generating {'ragged' if args.get('fixed_seq_len') is None else 'dense'} tensor inputs...")
    sequence_inputs = tf.ragged.constant(numericalized, dtype=tf.int32)
    subword_labels = tf.ragged.constant(labels, dtype=tf.int32)
    if args.get('fixed_seq_len') is not None:
        sequence_inputs = sequence_inputs.to_tensor(1) # padding symbol is 1 with ULMFiT, not 0!
        subword_labels = subword_labels.to_tensor(0)
        sequence_inputs = tf.keras.preprocessing.sequence.pad_sequences(sequence_inputs, maxlen=args['fixed_seq_len'],
                                                                        padding='post', truncating='post', value=1, dtype=int)
        subword_labels = tf.keras.preprocessing.sequence.pad_sequences(subword_labels, maxlen=args['fixed_seq_len'],
                                                                       padding='post', truncating='post', value=0, dtype=int)
    ulmfit_tagger, hub_object = ulmfit_sequence_tagger(model_type=args['model_type'],
                                                       pretrained_encoder_weights=args['model_weights_cp'],
                                                       spm_model_args=spm_args,
                                                       fixed_seq_len=args.get('fixed_seq_len'),
                                                       num_classes=args['num_classes'])
    ulmfit_tagger.summary()
    print(f"Shapes - sequence inputs: {sequence_inputs.shape}, labels: {subword_labels.shape}")
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = RaggedSparseCategoricalCrossEntropy() if args.get('fixed_seq_len') is None \
                                                    else tf.keras.losses.SparseCategoricalCrossentropy()

    # ##### This works only with fixed-length sequences:
    # ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    #                 filepath = args['out_cp_name'],
    #                 save_weights_only=True,
    #                 save_freq=25,
    #                 monitor='sparse_categorical_accuracy',
    #                 mode='auto',
    #                 save_best_only=True)
    # ulmfit_tagger.compile(optimizer='adam', loss=loss_fn, metrics=['sparse_categorical_accuracy'])
    # ulmfit_tagger.fit(sequence_inputs, subword_labels, epochs=1, batch_size=args['batch_size'],
    #                   callbacks=[ckpt_cb])

    ##### For RaggedTensors and variable-length sequences we have to use the GradientTape ########
    ulmfit_tagger.compile(optimizer='adam', loss=loss_fn, metrics=['sparse_categorical_accuracy'])
    batch_size = args['batch_size']
    steps_per_epoch = sequence_inputs.shape[0] // batch_size
    for epoch in range(args['num_epochs']):
        for step in range(steps_per_epoch - 1):
            if step % 25 == 0:
                print("Saving weights...")
                ulmfit_tagger.save_weights(args['out_cp_name'])
            train_step(model=ulmfit_tagger,
                       hub_object=hub_object,
                       loss_fn=loss_fn, optimizer=optimizer,
                       x=sequence_inputs[(step*batch_size):(step+1)*batch_size],
                       y=subword_labels[(step*batch_size):(step+1)*batch_size],
                       step_info=(step, steps_per_epoch))
        # TODO: add shuffling after every epoch
    return ulmfit_tagger, sequence_inputs, subword_labels, loss_fn, optimizer

if __name__ == "__main__":
    # TODO: the weights checkpoint quirk should be done away with, but to serialize anything custom into a SavedModel
    # especially if that thing contains RaggedTensors is kind of nightmarish...
    argz = argparse.ArgumentParser()
    argz.add_argument("--ddpl-iob", required=False, help="Training input file (assume whitespace tokenized)")
    argz.add_argument("--model-weights-cp", required=True, help="For training: path to *weights* (checkpoint) of " \
                                                                "the generic model (not the SavedModel/HDF5 blob!)." \
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
                                                          "3 classes will be used: 0 = 'O', 1 = 'B-N' and 2 = 'I-N'")
    argz.add_argument("--num-classes", type=int, default=3, help="Number of label categories")
    argz.add_argument("--out-cp-name", default="ulmfit_tagger", help="(Training only): Checkpoint name to save every 10 steps")
    argz = vars(argz.parse_args())
    if all([argz.get('max_seq_len') and argz.get('fixed_seq_len')]):
        print("You can use either `max_seq_len` with RaggedTensors to restrict the maximum sequence length, or `fixed_seq_len` with dense "\
              "tensors to set a fixed sequence length with automatic padding, not both.")
        exit(1)
    if argz.get('ddpl_iob') is None and argz.get('interactive') is None:
        print("Please provide either a data file for training / evaluation or run the script with --interactive switch")
        exit(0)
    if argz.get('interactive') is True:
        if argz.get('label_map') is not None:
            label_map = open(argz['label_map'], 'r', encoding='utf-8').readlines()
            label_map = {k:v.strip() for k,v in enumerate(label_map) if len(v)>0}
        else:
            label_map = DEFAULT_LABEL_MAP
        interactive_demo(argz, label_map)
    else:
        main(argz)
