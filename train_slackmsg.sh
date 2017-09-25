python slack_message_dataset.py --mapfile reaction_map.yml data/20170*.txt
python train.py --gpu 0 --input data/slack_comments.txt --label data/slack_labels.txt --seqlen 30 -e 20 -u 300
