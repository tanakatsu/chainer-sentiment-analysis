python predict.py --file data/slack_comments.txt --seqlen 30 -u 300 -n 2 --model data/rnnlm.model --input_vocab data/input_vocab.bin --label_vocab data/label_vocab.bin --label data/slack_labels.txt --fraction 0.9

### Verify by using training dataset ###
# python predict.py --file data/slack_comments.txt --label data/slack_labels.txt --seqlen 30 -u 300 -n 2 --model data/rnnlm.model --input_vocab data/input_vocab.bin --label_vocab data/label_vocab.bin --fraction 0.9


