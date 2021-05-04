import json
import argparse, readline
import tensorflow as tf
from ptools.lipytools.little_methods import r_jsonl
from modelling_scripts.ulmfit_tf2_heads import ulmfit_sequence_tagger
from modelling_scripts.ulmfit_tf2 import RaggedSparseCategoricalCrossEntropy
from lm_tokenizers import LMTokenizerFactory

DEFAULT_LABEL_MAP = {0: 'O', 1: 'B-N', 2: 'I-N'} # fixme: label map should not be hardcoded (maybe pass as parameter?)

def tokenize_and_align_labels(spmproc, ddpl_iob):
    """
    Performs Sentencepiece tokenization on an already whitespace-tokenized text
    and aligns labels to subwords
    """

    print(f"Tokenizing and aligning {len(ddpl_iob)} examples...")
    tokenized = []
    numericalized = []
    labels = []
    for sent in ddpl_iob:
        sentence_tokens = []
        sentence_ids = []
        sentence_labels = []
        sent = sent[:3000] # fixme: this should be a parameter. with bsize=32 we get GPU OOM errors
        for whitespace_token in sent:
            subwords = spmproc.encode_as_pieces(whitespace_token[0])
            sentence_tokens.extend(subwords)
            sentence_ids.extend(spmproc.encode_as_ids(whitespace_token[0]))
            sentence_labels.extend([whitespace_token[1]]*len(subwords))
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
    # spmproc = LMTokenizerFactory.get_tokenizer(tokenizer_type='spm', \
    #                                            tokenizer_file=args['spm_model_file'], \
    #                                            add_bos=True, add_eos=True) # bos and eos will need to be added manually
    ulmfit_tagger, spm_encoder = ulmfit_sequence_tagger(num_classes=len(label_map),
                                                        pretrained_weights=None,
                                                        spm_model_file=args['spm_model_file'],
                                                        also_return_spm_encoder=True)
    # Begin ugly hack - and only for demo!
    spmproc = spm_encoder.layers[-1].spmproc
    spmproc.add_bos = True
    spmproc.add_eos = True
    # End ugly hack

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

def train_step(model, loss_fn, optimizer, x, y, step_number): # todo: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit

    with tf.GradientTape() as tape:
        y_preds = model(x, training=True)
        loss_value = loss_fn(y_true=y, y_pred=y_preds)
        print(f"Step {step_number} | loss before applying gradients: {loss_value}")

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def main(args):
    ddpl_iob = r_jsonl(args['ddpl_iob'])
    spmproc = LMTokenizerFactory.get_tokenizer(tokenizer_type='spm', \
                                               tokenizer_file=args['spm_model_file'], \
                                               add_bos=False, add_eos=False) # bos and eos will need to be added manually
    tokenized, numericalized, labels = tokenize_and_align_labels(spmproc, ddpl_iob)
    print("Generating ragged tensor inputs...")
    sequence_inputs = tf.ragged.constant(numericalized, dtype=tf.int32)
    subword_labels = tf.ragged.constant(labels, dtype=tf.int32)
    #ulmfit_tagger = ulmfit_quick_and_dirty_sequence_tagger(num_classes=3, \
    #                                                       pretrained_weights=args['model_weights_cp'])

    ulmfit_tagger = ulmfit_sequence_tagger(num_classes=3,
                                           pretrained_weights=args['model_weights_cp'],
                                           spm_model_file=args['spm_model_file'],
                                           also_return_spm_encoder=False)
    ulmfit_tagger.summary()
    loss_fn_ragged = RaggedSparseCategoricalCrossEntropy()
    optimizer = tf.keras.optimizers.Adam()
    # ulmfit_tagger.compile(optimizer='adam', loss=loss_fn_ragged, metrics=['sparse_categorical_accuracy'])
    # ulmfit_tagger.compile(optimizer=optimizer, loss=loss_fn_ragged)
    # ulmfit_tagger.fit(sequence_inputs, subword_labels, epochs=1, batch_size=32)
    batch_size = args['batch_size']
    steps_per_epoch = sequence_inputs.shape[0] // batch_size
    for epoch in range(args['num_epochs']):
        for step in range(steps_per_epoch - 1):
            if step % 10 == 0:
                print("Saving weights...")
                ulmfit_tagger.save_weights(args['out_cp_name'])
            train_step(ulmfit_tagger,
                       loss_fn_ragged,
                       optimizer,
                       sequence_inputs[(step*batch_size):(step+1)*batch_size],
                       subword_labels[(step*batch_size):(step+1)*batch_size],
                       step)
    return ulmfit_tagger, sequence_inputs, subword_labels, loss_fn_ragged, optimizer

if __name__ == "__main__":
    # TODO: the weights checkpoint quirk should be done away with, but to serialize anything custom into a SavedModel
    # especially if that thing contains RaggedTensors is kind of nightmarish...
    argz = argparse.ArgumentParser()
    argz.add_argument("--ddpl-iob", required=False, help="Training input file (assume whitespace tokenized)")
    argz.add_argument("--model-weights-cp", required=True, help="For training: path to *weights* (checkpoint) of " \
                                                                "the generic model (not the SavedModel/HDF5 blob!)." \
                                                                "For demo: path to *weights* produced by this script")
    argz.add_argument('--spm-model-file', required=True, help="Path to SentencePiece model file")
    argz.add_argument("--batch-size", default=32, type=int, help="Batch size")
    argz.add_argument("--num-epochs", default=1, type=int, help="Number of epochs")
    argz.add_argument("--interactive", action='store_true', help="Run the script in interactive mode")
    argz.add_argument("--label-map", required=False, help="Path to a JSON file containing labels. If not given, " \
                                                          "a 3 classes will be used: 0 = 'O', 1 = 'B-N' and 2 = 'I-N'")
    argz.add_argument("--out-cp-name", default="ulmfit_tagger", help="(Training only): Checkpoint name to save every 10 steps")
    argz = vars(argz.parse_args())
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

    
