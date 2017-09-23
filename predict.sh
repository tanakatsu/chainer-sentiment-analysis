python predict.py --file data/reviews.2490-2500.txt --seqlen 200 -n 2 --model data/rnnlm.model --input_vocab data/input_vocab.bin --label_vocab data/label_vocab.bin

### Verify by using training dataset ###
# python predict.py --file data/reviews.txt --label data/labels.txt --seqlen 200 -n 2 --model data/rnnlm.model --input_vocab data/input_vocab.bin --label_vocab data/label_vocab.bin --fraction 0.9

