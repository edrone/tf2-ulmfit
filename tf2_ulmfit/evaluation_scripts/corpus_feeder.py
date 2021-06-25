from io import StringIO

import attr
import numpy as np
import tensorflow as tf

PAD_ID=1

@attr.s
class LMCorpusLoader:
    """
    Loads a language modelling corpus and generates training / evaluation sequences.

    :corpus_path:        sentence-tokenized data (plain text file, one line = one sentence)
    :is_numericalized:   True if `corpus_path` points to a file which is already numericalized.
                         If this is set to False, you must provide a path to an SPM model
                         in `spm_encoder`.
    :batch_size:         batch size
    :spm_encoder:        Path to an SPM model file. Should be set to None if `corpus_path` points
                         to a numericalized file.
    :max_seq_len:        Truncate sentences to this number of tokens.
    :padding_direction:  'pre' or 'post'
    """

    corpus_path = attr.ib()
    is_numericalized = attr.ib()
    batch_size = attr.ib(default=64)
    spm_encoder = attr.ib(default=None)
    max_seq_len = attr.ib(default=80)
    min_seq_len = attr.ib(default=10)
    padding_direction = attr.ib(default='post')

    def next_batch(self):
        """ Generates batches from a text file without slurping the entire data into memory """
        if self.is_numericalized is False and self.spm_encoder is None:
            raise ValueError("Please provide a path to SPM model file if your test corpus isn't converted "
                             "to token IDs")
        batch = []
        cnt = 0
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if self.is_numericalized is True:
                    tokens = np.genfromtxt(StringIO(line.strip()), dtype=int)
                else:
                    tokens = self.spm_encoder(tf.constant([line.strip()]))[0].numpy()
                if len(tokens) < self.min_seq_len: continue
                batch.append(tokens)
                cnt += 1
                if cnt == self.batch_size:
                    cnt = 0
                    if self.is_numericalized is True:
                        ret = tf.constant(batch, dtype=tf.int32)
                    else:
                        ret = tf.keras.preprocessing.sequence.pad_sequences(batch, value=PAD_ID,
                                                                            maxlen=self.max_seq_len,
                                                                            padding=self.padding_direction,
                                                                            truncating=self.padding_direction)
                    batch = []
                    print (ret)
                    yield ret


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