import os, copy
import attr
import numpy as np
from io import StringIO
import tensorflow as tf

PAD_ID=1

@attr.s
class LMCorpusLoader:
    """
    Loads a preprocessed corpus and generates training / evaluation sequences.

    Note: `corpus_path` points to sentence-tokenized data (plain text file, one line = one sentence)

    """
    corpus_path = attr.ib()
    skip_step = attr.ib(default=None)
    sliding_window = attr.ib(default=None)
    batch_size = attr.ib(default=64)
    tokenizer_obj = attr.ib(default=None)
    max_seq_len = attr.ib(default=80)
    min_seq_len = attr.ib(default=10)

    def next_batch_causal_lm(self, *, is_pretokenized=True, padding_direction='pre'): # this assumes tokenizer_obj is a SentencePieceProcessor
        """ Generates batches from a text file without slurping the entire data into memory """
        if is_pretokenized is False and self.tokenizer_obj is None:
            raise ValueError("Please provide a path to SPM model file if your test corpus isn't converted " \
                             "to token IDs")
        batch = []
        cnt = 0
        if is_pretokenized:
            pad_id = 1
        else:
            pad_id = self.tokenizer_obj.id_to_piece(self.tokenizer_obj.pad_id())
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if is_pretokenized is True:
                    tokens = np.genfromtxt(StringIO(line.strip()))
                else:
                    tokens = self.tokenizer_obj.encode_as_ids(line.strip())
                if len(tokens) < self.min_seq_len: continue
                batch.append(tokens)
                cnt += 1
                if cnt == self.batch_size:
                    cnt = 0
                    ret = tf.keras.preprocessing.sequence.pad_sequences(batch, value=pad_id, maxlen=self.max_seq_len, \
                                                                        padding=padding_direction, truncating=padding_direction)
                    batch = []
                    yield ret

    def pppl_sents_generator(self):
        """
        X: ['ala', 'ma', 'kota'] --> Y: [['<mask>', 'ala', 'kota'], ['ala', '<mask>', 'kota'], ['ala', 'ma', '<mask>']]"
        """
        fsize = os.path.getsize(self.corpus_path)
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                sent = self.tokenizer_obj.encode(line.strip(), add_special_tokens=True)
                sent = sent[0:768] # fixme: do not hardcode max seq len
                print(sent)
                if len(sent) < self.min_seq_len: continue
                masked_sents = []; masked_tokens = []
                for i in range(1, len(sent)-1): # sliding window from first to penultimate token (0 = <s> and last = </s>, so these are left untouched)
                    masked_sents.append(copy.deepcopy(sent))
                    masked_sents[-1][i] = self.tokenizer_obj.mask_token_id
                    masked_tokens.append(sent[i])
                yield np.array(masked_sents), np.array(masked_tokens)
        # raise NotImplementedError("X: ['ala', 'ma', 'kota'] --> Y: ['ala', '<mask>', 'kota']")
    
def restore_model(*, model_type, pretrained_path):
    """ Sadly we can't assume `model` to be an instance of tf.keras.model - Polish Roberta
        gets exported to a TFAutoModelForMaskedLM *almost* correctly by HuggingFace transformers.
        There is still one bias matrix which doesn't get loaded and so we are forced to use
        the PyTorch implementation. The world of deep learning is weird at times.
    """
    if model_type == 'causal':
        model = tf.keras.models.load_model(pretrained_path)
    elif model_type == 'polish_roberta':
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM(pretrained_path)
        model.eval()
        raise NotImplementedError("Not yet")
    else:
        raise ValueError(f"Unsupported model type {model_type}")
    return model

# todo: find a more clever way of handling this
def modify_max_seq_len(*, pretrained_model, new_max_seq_len): # fixme - this only works with one particular NN setup and is memory-inefficient
    """ Modifies a pretrained model's input sequence length.
        This allows you to control BPTT window and has no effect on training.
    """
    orig_vocab_size = pretrained_model.output_shape[-1]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=PAD_ID, input_shape=(new_max_seq_len, )))
    model.add(tf.keras.layers.Embedding(orig_vocab_size, 400,
                                        input_length=new_max_seq_len))
    model.add(tf.keras.layers.LSTM(1024, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.TimeDistributed(
                              tf.keras.layers.Dense(orig_vocab_size, activation='softmax')))
    model.set_weights(pretrained_model.get_weights())
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    return model

def predict_all_arch(*, model_type, model, x_data):
    """ Model-independent predict method (not used currently) """
    if model_type == 'causal':
        return model(x_data)
    elif model_type == 'polish_roberta':
        import torch
        pt_tensors = torch.tensor(x_data)
        ret = model(pt_tensors)[0].detach().numpy()
        return ret
    else:
        raise ValueError(f"Unknown model type {model_type}")

def tensor_shift(*, data, positions, axis, pad_fill):
    """ Shifts all tensor values by a number of positions to the left/right.

        Essentially does the same thing as tf.roll, but without wrapping.
    """
    shifted = tf.roll(data, positions, axis).numpy()
    if positions == -1:
        shifted[:, -1] = pad_fill
    elif positions < -1:
        shifted[:, positions:] = pad_fill
    elif positions > 0:
        shifted[:, 0:positions] = pad_fill
    return shifted
