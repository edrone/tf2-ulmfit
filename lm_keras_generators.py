import os
import numpy as np
import tensorflow as tf
from io import StringIO

class KerasLMSentenceLevelBatchGenerator(object):
    """ Adapted from http://adventuresinmachinelearning.com/keras-lstm-tutorial/
    
        Does not use continous / running text as the original implementation, but rather
        text that has been sentence-tokenized in advance.
    
        Given:
        [['<pad>', '<pad>', '<s>', 'The', 'cat', 'sat', 'on', 'a', 'mat', 'and', 'ate', 'his', 'hat', '.', '<eos>']]
    
        and: skip_steps=1
    
        Yield:
        [
            ['<pad>', '<pad>', '<s>', 'The', 'cat'],
            ['<pad>', '<s>', 'The', 'cat', 'sat'],
            ['<s>', 'The', 'cat', 'sat', 'on'],
            ['The', 'cat', 'sat', 'on', 'a'],
            ['cat', 'sat', 'on', 'a', 'mat'],
            ['on', 'a', 'mat', 'and', 'ate'],
            ['a', 'mat', 'and', 'ate', 'his'],
            ['mat', 'and', 'ate', 'his', 'hat'],
            ['and', 'ate', 'his', 'hat', '.'],
        ],
        [
            ['<pad>', '<s>', 'The', 'cat', 'sat'],
            ['<s>', 'The', 'cat', 'sat', 'on'],
            ['The', 'cat', 'sat', 'on', 'a'],
            ['cat', 'sat', 'on', 'a', 'mat'],
            ['on', 'a', 'mat', 'and', 'ate'],
            ['a', 'mat', 'and', 'ate', 'his'],
            ['mat', 'and', 'ate', 'his', 'hat'],
            ['and', 'ate', 'his', 'hat', '.'],
            ['ate', 'his', 'hat', '.', '<eos>'],
        ],
    
    
        We expect x_sequences to be padded, but they don't need to be arrays of strings. In fact, it's
        more likely that we will see arrays of indices.
    """

    def __init__(self, *, x_sequences, max_seq_len, min_seq_len, num_shifted_sentences, pad_idx_or_symbol, skip_step=5, \
                 explicit_x_seq_len=None, explicit_batch_size=None, strategy='shift_as_needed'):
        if skip_step > max_seq_len:
            raise ValueError("Skip step needs to be greater than or equal to the max sequence length")
        self.x_sequences = x_sequences
        if strategy in ['slurp', 'slurp_as_is']:
            self.x_sequences = []
            with open(x_sequences, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = np.genfromtxt(StringIO(line)) if strategy == 'slurp' else list(map(int, line.split()))
                    if len(tokens) > min_seq_len: self.x_sequences.append(tokens)
            if strategy == 'slurp':
                self.x_sequences = tf.keras.preprocessing.sequence.pad_sequences(self.x_sequences, padding='post', truncating='post',\
                                                                                 maxlen=max_seq_len, \
                                                                                 value=pad_idx_or_symbol)
            self.explicit_seq_len = len(x_sequences)
        else:
            self.explicit_seq_len = explicit_x_seq_len
        self.explicit_batch_size = explicit_batch_size
        self.strategy = strategy
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.num_shifted_sentences = num_shifted_sentences
        # self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step
        self.pad_idx_or_symbol = pad_idx_or_symbol
        self.unk_symbol_idx = 0

    def get_num_sliding_windows(self):
        return self.max_seq_len // self.skip_step # each batch will generate num_shifted_sentences * num_sliding_windows examples

    def get_batch_size(self):
        if self.explicit_batch_size is None:
            return self.num_shifted_sentences*self.get_num_sliding_windows()
        else:
            return self.explicit_batch_size

    def get_epoch_size(self):
        return self.explicit_seq_len*self.get_num_sliding_windows()

    def get_steps_per_epoch(self):
        return self.get_epoch_size() // self.get_batch_size()

    def print_batch_info(self, batch_size=None):
        cols = "{0:8}{1:40}{2:6}"
        print("******************** BATCH GENERATION SUMMARY **************************")
        print(cols.format("  (1)", "Total # of sentences;", self.explicit_seq_len))
        print(cols.format("  (2)", "Max sequence length", self.max_seq_len))
        print(cols.format("  (3)", "Skip step", self.skip_step))
        print(cols.format("  (4)", "# of sliding windows per sentence", self.get_num_sliding_windows()))
        print(cols.format("  (5)", "# sentences to shift in each batch", self.num_shifted_sentences))
        print(cols.format("  (6)", "# of examples per epoch", self.get_epoch_size()))
        print(cols.format("  (7)", "Batch size", self.get_batch_size()))
        print(cols.format("  (8)", "# of steps per epoch", self.get_steps_per_epoch()))
        print("************************************************************************")

    def generate(self, **kwargs):
        if self.strategy == 'shift_as_needed':
            return self._generate_shift_as_needed(kwargs)
        elif self.strategy == 'slurp':
            return self._generate_slurped()
        elif self.strategy == 'slurp_as_is':
            raise ValueError("Generating sequences is not available for `slurp_as_is`. " \
                             "Please access the x_sequences field directly instead.")
        elif self.strategy == 'from_disk':
            return self._generate_from_disk()
        elif self.strategy == 'running_text':
            return self._generate_continuous()
        else:
            raise ValueError(f"Unknown batch generation strategy {self.strategy}")

    def _generate_slurped(self):
        num_sliding_windows = self.max_seq_len // self.skip_step # each batch will generate num_shifted_sentences * num_sliding_windows examples
        print(f"SLURP / Number of sliding windows: {num_sliding_windows}")
        print(f"SLURP / Expected number of rows in shifted x_sequences: {num_sliding_windows*self.num_shifted_sentences}")

        x = []
        y = []
        while True:
            if self.current_idx + self.num_shifted_sentences >= self.explicit_seq_len:
                # reset the index back to the start of the data set
                self.current_idx = 0
            for window_idx in range(num_sliding_windows): # shift the sequence *to the left* by `skip_step` tokens
                x_shifted = self.x_sequences[self.current_idx:self.current_idx+self.num_shifted_sentences][:,window_idx*self.skip_step:]
                for single_row in x_shifted: # show only sequences that contain something else than padding
                    if np.all(single_row == np.full_like(single_row, self.pad_idx_or_symbol)):
                        continue
                    x.append(single_row)
                    y.append(single_row[1:])
                    if len(x) == len(y) == self.explicit_batch_size:
                        x_arr = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=self.max_seq_len, value=1, padding="post")
                        y_arr = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=self.max_seq_len, value=1, padding="post")
                        x = []
                        y = []
                        yield x_arr, y_arr
            self.current_idx += self.num_shifted_sentences

    def _generate_from_disk(self):
        num_sliding_windows = self.max_seq_len // self.skip_step # each batch will generate num_shifted_sentences * num_sliding_windows examples
        print(f"DISK / Number of sliding windows: {num_sliding_windows}")
        print(f"DISK / Expected number of rows in shifted x_sequences: {num_sliding_windows*self.num_shifted_sentences}")
        fsize = os.path.getsize(self.x_sequences)
        fd = open(self.x_sequences, 'r', encoding='utf-8')
        while True:
            x_local_sequences = []
            cnt = 0
            #for i in range(self.num_shifted_sentences):
            while True:
                if fd.tell() > fsize: fd.seek(0)
                line = np.genfromtxt(StringIO(fd.readline()), dtype=int)
                if len(line) < self.min_seq_len: continue
                if len(line) > self.max_seq_len: line[self.max_seq_len-1] = 3
                x_local_sequences.append(line)
                cnt += 1
                if cnt >= self.num_shifted_sentences: break
            x_local_sequences = tf.keras.preprocessing.sequence.pad_sequences(x_local_sequences, \
                                                                              padding='post', truncating='post',\
                                                                              maxlen=self.max_seq_len, \
                                                                              value=self.pad_idx_or_symbol)
            x = []
            y = []

            for window_idx in range(num_sliding_windows): # shift the sequence *to the left* by `skip_step` tokens
                x_shifted = x_local_sequences[:][:,window_idx*self.skip_step:]
                x.extend(x_shifted)
                y.extend(x_shifted[:, 1:])
            x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=self.max_seq_len, value=1, padding="post")
            y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=self.max_seq_len, value=1, padding="post")
            yield x, y

    def _generate_shift_as_needed(self, return_piece_ids=True, pad_symbol=1, remove_unks=True, explicit_batch_size=64):
        print(f"DISK / Shift-as-needed")
        fsize = os.path.getsize(self.x_sequences)
        fd = open(self.x_sequences, 'r', encoding='utf-8')
        x = []
        y = []
        while True:
            x_local_sequences = []
            cnt = 0
            #for i in range(self.num_shifted_sentences):
            while True:
                if fd.tell() >= fsize: fd.seek(0) # wrap around if EOF reached: this ensures batches of fixed size
                line = np.genfromtxt(StringIO(fd.readline()), dtype=int).tolist()
                if remove_unks:
                    line = [t for t in line if t != self.unk_symbol_idx]
                if len(line) < self.min_seq_len: continue
                x_local_sequences.append(line)
                cnt += 1
                if cnt >= self.num_shifted_sentences: break
            x_local_sequences = tf.keras.preprocessing.sequence.pad_sequences(x_local_sequences, \
                                                                              padding='post', \
                                                                              value=pad_symbol)
            for window_idx in range((max(self.max_seq_len, (x_local_sequences.shape[1] - self.max_seq_len)) // self.skip_step) + 1):
                x_shifted = x_local_sequences[:, window_idx*self.skip_step:(window_idx*self.skip_step)+self.max_seq_len]
                if np.all(x_shifted == np.full_like(x_shifted, pad_symbol)): # no point in left shifting if all we see is padding
                    break
                for single_row in x_shifted: # show only sequences that contain something else than padding
                    if np.all(single_row == np.full_like(single_row, pad_symbol)):
                        continue
                    x.append(single_row)
                    y.append(single_row[1:])
                    if len(x) == len(y) == explicit_batch_size:
                        x_arr = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=self.max_seq_len, value=1, padding="post")
                        y_arr = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=self.max_seq_len, value=1, padding="post")
                        x = []
                        y = []
                        yield x_arr, y_arr

                #x.extend(x_shifted)
                #y.extend(x_shifted[:, 1:])

                #if len(x) == len(y) > true_bsize:
                #    x_arr = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=self.max_seq_len, value=1, padding="post")
                #    y_arr = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=self.max_seq_len, value=1, padding="post")
                #    x = []
                #    y = []
                #    yield x_arr, y_arr

    def _generate_continuous(self):
        raise NotImplementedError("WIP")
        num_sliding_windows = self.max_seq_len // self.skip_step # each batch will generate num_shifted_sentences * num_sliding_windows examples
        print(f"DISK-CONT / Number of sliding windows: {num_sliding_windows}")
        print(f"DISK-CONT / Expected number of rows in shifted x_sequences: {num_sliding_windows*self.num_shifted_sentences}")
        fsize = os.path.getsize(self.x_sequences)
        fd = open(self.x_sequences, 'r', encoding='utf-8')
        while True:
            x_local_sequences = []
            cnt = 0
            #for i in range(self.num_shifted_sentences):
            while True:
                if fd.tell() > fsize: fd.seek(0)
                nums = ""
                sequence_counter = 0
                while True:
                    c = fd.read(1)
                    nums += c
                    if c in {' ', '\n'}:
                        cnt += 1
                    if cnt > 80:
                        break

                line = np.genfromtxt(StringIO(fd.readline()), dtype=int)
                if len(line) < self.min_seq_len: continue
                if len(line) > self.max_seq_len: line[self.max_seq_len-1] = 3
                x_local_sequences.append(line)
                cnt += 1
                if cnt >= self.num_shifted_sentences: break
            x_local_sequences = tf.keras.preprocessing.sequence.pad_sequences(x_local_sequences, \
                                                                              padding='post', truncating='post',\
                                                                              maxlen=self.max_seq_len, \
                                                                              value=self.pad_idx_or_symbol)
            x = []
            y = []

            for window_idx in range(num_sliding_windows): # shift the sequence *to the left* by `skip_step` tokens
                x_shifted = x_local_sequences[:][:,window_idx*self.skip_step:]
                x.extend(x_shifted)
                y.extend(x_shifted[:, 1:])
            x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=self.max_seq_len, value=1, padding="post")
            y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=self.max_seq_len, value=1, padding="post")
            yield x, y

    def reset(self):
        self.current_idx = 0
