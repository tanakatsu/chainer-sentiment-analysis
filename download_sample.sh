wget -O data/labels.txt https://raw.githubusercontent.com/udacity/deep-learning/master/sentiment-network/labels.txt
wget -O data/reviews.txt https://raw.githubusercontent.com/udacity/deep-learning/master/sentiment-network/reviews.txt

tail -n 10 data/reviews.txt > data/reviews.2490-2500.txt
# expected labels  are     positive, negative, positive, negative, positive, negative, positive, negative, positive, negative
# predicted labels may be  positive, positive, positive, negative, positive, negative, positive, negative, negative, negative
