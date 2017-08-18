#!/usr/bin/env python
"""Example to generate text from a recurrent neural network language model.

This code is ported from following implementation.
https://github.com/longjie/chainer-char-rnn/blob/master/sample.py

"""
import argparse
import sys

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import serializers

import net
import dataset


def iconv(int_data, int_to_vocab):
    return [int_to_vocab[x] for x in int_data if not x == 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='model data, saved by train.py')
    parser.add_argument('--file', '-f', type=str, required=True,
                        help='input text file, used for reaction analysis')
    parser.add_argument('--label', type=str, default=None,
                        help='label file for calculating accuracy')
    parser.add_argument('--input_vocab', '-i', type=str, default='data/input_vocab.bin',
                        help='input text vocaburaly dictionary')
    parser.add_argument('--label_vocab', '-l', type=str, default='data/label_vocab.bin',
                        help='input label vocaburaly dictionary')
    parser.add_argument('--seqlen', type=int, required=True,
                        help='sequence length')
    parser.add_argument('-n', type=int, default=3,
                        help='number of candidates')
    parser.add_argument('--fraction', type=float, default=0.0,
                        help='split ratio of dataset (0 means all data goes to test)')
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='number of units')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')

    args = parser.parse_args()
    xp = cuda.cupy if args.gpu >= 0 else np

    # load dataset and vocabulary
    seq_len = args.seqlen

    # For testing with labels
    # ds = dataset.Dataset(args.file, label=args.label, input_vocab=args.input_vocab, label_vocab=args.label_vocab, seq_len=seq_len)

    ds = dataset.Dataset(args.file, label=args.label, input_vocab=args.input_vocab, label_vocab=args.label_vocab, seq_len=seq_len, fraction=args.fraction)
    _, test = ds.get_inputs_and_labels()

    input_vocab, label_vocab = ds.get_vocab()

    input_ivocab = {i: c for c, i in input_vocab.items()}
    label_ivocab = {i: c for c, i in label_vocab.items()}

    # should be same as n_units, described in train.py
    n_units = args.unit

    lm = net.RNNLM(len(input_vocab), len(label_vocab), n_units, train=False)
    model = L.Classifier(lm)

    serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    n_top = args.n
    n_match = 0
    n_total = 0
    has_label = (len(test[0]) == len(test[1]))
    sys.stdout.write('\n')

    for i, data in enumerate(test[0]):
        print('[input {0}/{1}]'.format(i + 1, len(test[0])), ' '.join(iconv(data, input_ivocab)))

        model.predictor.reset_state()
        for j in six.moves.range(seq_len):
            word = chainer.Variable(xp.array([data[j]]), volatile='on')
            pred = F.softmax(model.predictor(word))
            if j == seq_len - 1:
                if args.gpu >= 0:
                    pred_data = cuda.to_cpu(pred.data)
                else:
                    pred_data = pred.data
                indice = pred_data[0].argsort()[-n_top:][::-1]
                probs = pred_data[0][indice]

                result = [(label_ivocab[idx], prob) for (idx, prob) in zip(indice, probs)]
                if has_label:
                    y = test[1][i]
                    print('[suggested reactions] %s' % result)
                    n_total += 1
                    if indice[0] == y:
                        print(label_ivocab[indice[0]], '(prediction)', '==', label_ivocab[y], '(actual)', '? => MATCH')
                        n_match += 1
                    else:
                        print(label_ivocab[indice[0]], '(prediction)', '==', label_ivocab[y], '(actual)', '? => NOT MATCH')
                else:
                    print('[suggested reactions] %s' % result)
        if has_label:
            print('cumulative accuracy=%f' % (n_match / n_total))
        sys.stdout.write('\n')


if __name__ == '__main__':
    main()
