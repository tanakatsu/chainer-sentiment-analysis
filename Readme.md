# Chainer sentiment analysis

## How to use

### Prepare dataset

##### Sample dataset

```
$ ./download_dataset.sh
```
Downloaded files are `data/reviews.txt` and `data/labels.txt`.

##### Your own dataset

You create two files, input text data file and labels data file.
Each file has one data (sentence or label) in each line.

### Training

```
$ python train.py --input data/reviews.txt --label data/labels.txt --seqlen 200 -e 3
```
##### parameters

- --input : input text file
- --label : label file
- --epoch or --e : number of epochs to learn
- --seqlen : sequence length
- --unit or -u : number of units 
- --batchsize or -b : learning minibatch size

##### output files

- data/input\_vocab.bin
- data/label\_vocab.bin
- data/rnnlm.model


### Prediction

```
$ python predict.py --file data/reviews.2490-2500.txt --seqlen 200 --model data/rnnlm.model --input_vocab data/input_vocab.bin --label_vocab data/label_vocab.bin
```

##### parameters

- --file : text file to analyze sentiment
- --model : model data saved by train.py
- --input_vocab : input text vocabulary file saved by train.py
- --label_vocab : label vocabulary file saved by train.py
- --seqlen : sequence length
- --unit or -u : number of units 

You have to use same parameters as in training for `seqlen` and `unit`.

Also, you have to use output files in training for `model` and `input_vocab`, `label_vocab` parameters.

## License

MIT
