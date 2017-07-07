#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import print_function
import argparse
import math
import sys
import time

import numpy as np
import six

import six.moves.cPickle as pickle

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import net
import dataset
import os


parser = argparse.ArgumentParser()
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=10, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=650, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=32,
                    help='learning minibatch size')
parser.add_argument('--seqlen', '-l', type=int, default=200,
                    help='sequence length')
parser.add_argument('--gradclip', '-c', type=int, default=5,
                    help='gradient norm threshold to clip')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--check', dest='check', action='store_true')
parser.add_argument('--input', '-i', default='data/reviews.txt',
                    help='input text file')
parser.add_argument('--label', default='data/labels.txt.',
                    help='label file')
parser.set_defaults(test=False)

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.epoch   # number of epochs
n_units = args.unit  # number of units per layer
batchsize = args.batchsize   # minibatch size
seq_len = args.seqlen   # sequence length
grad_clip = args.gradclip    # gradient norm threshold to clip

ds = dataset.Dataset(args.input, label=args.label, seq_len=seq_len)

# Load data
train, test = ds.get_inputs_and_labels()
print('#data size =', train[0].shape, train[1].shape)

# Get vocabulary dictionary
input_vocab, label_vocab = ds.get_vocab()
print('#vocab size =', len(input_vocab), len(label_vocab))

if args.test:
    print('TEST mode!')
    train = (train[0][0:100], train[1][0:100])
    test = (test[0][0:100], test[1][0:100:])
    print(train[0].shape)

with open(os.path.join(os.path.dirname(args.input), 'input_vocab.bin'), 'wb') as f:
    pickle.dump(input_vocab, f)
with open(os.path.join(os.path.dirname(args.input), 'label_vocab.bin'), 'wb') as f:
    pickle.dump(label_vocab, f)
print('save vocaburay dictionary')

# Prepare RNNLM model, defined in net.py
lm = net.RNNLM(len(input_vocab), len(label_vocab), n_units)
model = L.Classifier(lm)
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
#optimizer = optimizers.SGD(lr=1.)
optimizer = optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)


def evaluate(dataset):
    # Evaluation routine
    evaluator = model.copy()  # to use different state
    evaluator.predictor.reset_state()  # initialize state
    evaluator.predictor.train = False  # dropout does nothing

    sum_log_perp = 0
    sum_accuracy = 0
    jump = len(dataset[0]) // batchsize

    for i in six.moves.range(0, jump * batchsize, batchsize):
        evaluator.predictor.reset_state()  # initialize state
        x_batch = dataset[0][i:i + batchsize]
        t_batch = dataset[1][i:i + batchsize]
        for j in range(seq_len):
            x = chainer.Variable(xp.asarray([x_batch[i][j] for i in range(batchsize)]), volatile='on')
            t = chainer.Variable(xp.asarray([t_batch[i] for i in range(batchsize)]), volatile='on')
            loss = evaluator(x, t)
            if j == seq_len - 1:
                evaluator.accuracy.to_cpu()
                sum_log_perp += loss.data
                sum_accuracy += evaluator.accuracy.data
    return math.exp(float(sum_log_perp) / jump), sum_accuracy / jump


def predict(dataset):
    evaluator = model
    evaluator.predictor.reset_state()  # initialize state
    evaluator.predictor.train = False  # dropout does nothing

    count = 0
    match = 0
    jump = len(dataset[0])

    for i in six.moves.range(jump):
        evaluator.predictor.reset_state()  # initialize state
        x_data = dataset[0][i]
        for j in range(seq_len):
            x = chainer.Variable(xp.asarray([x_data[j]]), volatile='on')
            pred = F.softmax(model.predictor(x))
            if j == seq_len - 1:
                if args.gpu >= 0:
                    pred_data = cuda.to_cpu(pred.data)
                else:
                    pred_data = pred.data
                indice = pred_data[0].argsort()[-1:][::-1]
                y = dataset[1][i]
                count += 1
                if indice[0] == y:
                    match += 1
    return match / count

# Learning loop
whole_len = train[0].shape[0]
print('whole_len=', whole_len)
seq_len = train[0].shape[1]
print('seq_len=', seq_len)
jump = whole_len // batchsize
cur_log_perp = xp.zeros(())
epoch = 0
start_at = time.time()
cur_at = start_at
accum_loss = 0
print('going to train {} iterations'.format(jump * n_epoch))

iteration = 0
for epoch in six.moves.range(1, n_epoch + 1):
    # print('epoch', epoch)
    for i in six.moves.range(0, jump * batchsize, batchsize):
        iteration += 1
        x_batch = train[0][i:i + batchsize]
        t_batch = train[1][i:i + batchsize]
        for j in range(seq_len):
            x = chainer.Variable(xp.asarray([x_batch[i][j] for i in range(batchsize)]))
            t = chainer.Variable(xp.asarray([t_batch[i] for i in range(batchsize)]))
            loss_i = model(x, t)

            if j == seq_len - 1:  # Run truncated BPTT
                # Last sequence output matters
                accum_loss += loss_i
                cur_log_perp += loss_i.data
                if iteration % 5 == 0:
                    print('epoch=', epoch, 'iteration=', iteration, 'loss=', loss_i.data)

                model.zerograds()
                accum_loss.backward()
                accum_loss.unchain_backward()  # truncate
                accum_loss = 0
                optimizer.update()
                model.predictor.reset_state()  # initialize state
            sys.stdout.flush()

        if iteration % 1000 == 0:
            now = time.time()
            throuput = 10000. / (now - cur_at)
            perp = math.exp(float(cur_log_perp) / 1000)
            print('iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(
                iteration, perp, throuput))
            cur_at = now
            cur_log_perp.fill(0)

    print('evaluate...')
    now = time.time()
    perp, acc = evaluate(test)
    print('epoch {} validation perplexity: {:.2f} validation accuracy: {:.2f}'.format(epoch, perp, float(acc)))  # acc = <class 'cupy.core.core.ndarray'>
    cur_at += time.time() - now  # skip time of evaluation

    # if epoch >= 6:
    #     optimizer.lr /= 1.2
    #     print('learning rate =', optimizer.lr)

# Evaluate on test dataset
print('test')
test_perp, test_acc = evaluate(test)
print('test perplexity:', test_perp)
print('test accuracy:', test_acc)

if args.check:
    test_acc = predict(test)
    print('test accuracy(check1):', test_acc)

# Save the model and the optimizer
print('save the model')
serializers.save_npz(os.path.join(os.path.dirname(args.input), 'rnnlm.model'), model)

if args.check:
    print('load the model')
    serializers.load_npz(os.path.join(os.path.dirname(args.input), 'rnnlm.model'), model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    test_acc = predict(test)
    print('test accuracy(check2):', test_acc)
