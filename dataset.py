from collections import Counter
from argparse import ArgumentParser
import numpy as np
from string import punctuation
import six.moves.cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer


class Dataset:

    def __init__(self, input, label=None, seq_len=None, fraction=0.9, input_vocab=None, label_vocab=None,
                 input_vocab_size=None, label_vocab_size=None, max_df=1.0, min_df=1):
        self.input_file = input
        self.label_file = label
        self.seq_len = seq_len
        self.fraction = fraction
        self.input_vocab = input_vocab
        self.label_vocab = label_vocab
        self.input_vocab_size = input_vocab_size
        self.label_vocab_size = label_vocab_size
        self.max_df = max_df
        self.min_df = min_df

    def get_inputs_and_labels(self):
        self.inputs, self.labels = self.read_data()
        print('read inputs and labels:', 'inputs=', len(self.inputs), 'labels=', len(self.labels))

        # build vocabulary dictionary
        self.build_vocab()
        print('build vocabulary:', 'inputs=', len(self.input_conv), 'labels=', len(self.label_conv))

        # preprocess inputs
        self.preprocess_inputs(self.max_df, self.min_df, self.input_vocab_size)

        # preprocess labels
        self.preprocess_labels(self.label_vocab_size)

        # reconstruct inputs and labels
        self.reconstruct_inputs_and_labels()

        # reconstruct vocabulary
        self.build_vocab()
        print('rebuild vocabulary:', 'inputs=', len(self.input_conv), 'labels=', len(self.label_conv))

        # encode
        self.encode()
        print('encoded:', 'train=', self.enc_inputs.shape, 'test=', self.enc_labels.shape)

        # split into train and test
        train_inputs, test_inputs = self.split_data(self.enc_inputs, self.fraction)
        train_labels, test_labels = self.split_data(self.enc_labels, self.fraction)
        train = (train_inputs, train_labels)
        test = (test_inputs, test_labels)
        print('split dataset:', 'train=', train[0].shape[0], 'test=', test[0].shape[0])

        return train, test

    def read_data(self):
        with open(self.input_file, 'r') as f:
            inputs = f.readlines()
            inputs = [x.rstrip() for x in inputs]
        if self.label_file:
            with open(self.label_file, 'r') as f:
                labels = f.readlines()
                labels = [x.rstrip() for x in labels]
        else:
            labels = []

        inputs_no_punctuation = []
        for text in inputs:
            inputs_no_punctuation.append(''.join([c for c in text if c not in punctuation]))
        inputs = inputs_no_punctuation
        return inputs, labels

    def build_vocab(self):
        if not self.input_vocab:
            all_text = ' '.join(self.inputs)
            words = all_text.split()
            self.input_counts = Counter(words)
            del self.input_counts['<PAD>']
            _input_vocab = sorted(self.input_counts, key=self.input_counts.get, reverse=True)
            self.input_conv = {c: i for i, c in enumerate(_input_vocab, 1)}
            self.input_conv['<PAD>'] = 0
            self.input_iconv = {}
            for k, v in self.input_conv.items():
                self.input_iconv[v] = k
        else:
            with open(self.input_vocab, mode='rb') as f:
                self.input_conv = pickle.load(f)
            self.input_iconv = {}
            for k, v in self.input_conv.items():
                self.input_iconv[v] = k

        if not self.label_vocab:
            self.label_counts = Counter(self.labels)
            label_vocab = sorted(self.label_counts, key=self.label_counts.get, reverse=True)
            self.label_conv = {c: i for i, c in enumerate(label_vocab)}
            self.label_iconv = {}
            for k, v in self.label_conv.items():
                self.label_iconv[v] = k
        else:
            with open(self.label_vocab, mode='rb') as f:
                self.label_conv = pickle.load(f)
            self.label_iconv = {}
            for k, v in self.label_conv.items():
                self.label_iconv[v] = k

    def preprocess_inputs(self, max_df, min_df, input_vocab_size):
        # remove words by tf-idf
        vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b', min_df=min_df, max_df=max_df)
        vectorizer.fit_transform(self.inputs)
        feature_names = set(vectorizer.get_feature_names())
        _inputs = []
        for input in self.inputs:
            _inputs.append(' '.join([x for x in input.split() if x in feature_names]))
        self.inputs = _inputs

        # remove least common words
        if input_vocab_size:
            most_common_keys = [x[0] for x in self.input_counts.most_common(input_vocab_size)]
            unknown_words = set(self.input_counts.keys()) - set(most_common_keys)

            _inputs = []
            for input in self.inputs:
                _inputs.append(' '.join([x for x in input.split() if x not in unknown_words]))
            self.inputs = _inputs

        return self.inputs

    def preprocess_labels(self, label_vocab_size):
        if self.label_vocab_size:
            most_common_keys = [x[0] for x in self.label_counts.most_common(label_vocab_size)]
            unknown_labels = set(self.label_counts.keys()) - set(most_common_keys)

            self.labels = [x if x not in unknown_labels else None for x in self.labels]
        return self.labels

    def reconstruct_inputs_and_labels(self):
        _inputs = []
        _labels = []
        if self.label_file:
            for i, l in zip(self.inputs, self.labels):
                if l:  # exclude 'None'
                    _inputs.append(i)
                    _labels.append(l)
            self.inputs = _inputs
            self.labels = _labels

    def encode(self):
        inputs_int = []
        for text in self.inputs:
            inputs_int.append([self.input_conv[x] for x in text.split() if x in self.input_conv])
        labels_int = [self.label_conv[x] for x in self.labels]

        # remove size zero
        if not labels_int:
            inputs_int = [x for x in inputs_int if len(x) > 0]
        else:
            inputs_labels_int = [(i, l) for i, l in zip(inputs_int, labels_int) if len(i) > 0]
            inputs_int = [x[0] for x in inputs_labels_int]
            labels_int = [x[1] for x in inputs_labels_int]

        # padding
        if self.seq_len:
            seq_len = self.seq_len
        else:
            seq_len = max([len(l) for l in inputs_int])
        inputs_int_incl_pad = np.zeros((len(inputs_int), seq_len), dtype=int)
        for i, row in enumerate(inputs_int):
            inputs_int_incl_pad[i, -len(row):] = np.array(row)[:seq_len]

        self.enc_inputs = np.array(inputs_int_incl_pad, np.int32)
        self.enc_labels = np.array(labels_int, np.int32)

    def get_vocab(self):
        return self.input_conv, self.label_conv

    def split_data(self, data, frac):
        size = int(len(data) * frac)
        return data[0:size], data[size:]


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    dataset = Dataset(input='data/reviews.txt', label='data/labels.txt', seq_len=200)
    # dataset = Dataset(input='data/reviews.txt', seq_len=200, input_vocab='data/input_vocab.bin', fraction=1.0)
    train, test = dataset.get_inputs_and_labels()
