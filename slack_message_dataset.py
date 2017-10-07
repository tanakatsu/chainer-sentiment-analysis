import MeCab
import re
from argparse import ArgumentParser
from collections import Counter
import yaml


class WordExtracter:

    def __init__(self):
        pass

    def isascii(self, s):
        asciiReg = re.compile(r'^[!-~]+$')
        return asciiReg.match(s) is not None

    def split_word_by_mecab(self, s):
        mecab = MeCab.Tagger("-Ochasen")
        res = mecab.parse(s)
        words = [x.split('\t')[0] for x in res.split('\n') if x != '']
        return words[0:-1]  # remove EOS

    def split_word_by_emoji(self, s):
        words = []
        while True:
            m = re.search(':[a-z_-]+:', s)
            if not m:
                break
            if m.start() > 0:
                w = s[0:m.start()]
                if self.isascii(w):
                    words.append(w)
                else:
                    words.extend(self.split_word_by_mecab(w))
            words.append(m.group(0))
            s = s[m.end():]
        if not words:
            if self.isascii(s):
                words.extend([s])
            else:
                words.extend(self.split_word_by_mecab(s))
        return words

    def get_words(self, content):
        all_words = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            line_words = []
            words = line.split()
            for word in words:
                line_words.extend(self.split_word_by_emoji(word))
            all_words.append(line_words)
        return all_words


class SlackMessageDataset:

    def __init__(self, seq_len=None, reaction_map=None, even=False):
        self.word_extracter = WordExtracter()
        self.words = []
        self.seq_len = seq_len
        self.reaction_map = reaction_map
        self.even = even

    def add_file(self, filename):
        with open(filename, 'r') as f:
            content = f.read()
            words = self.word_extracter.get_words(content)
            # print(words)
        self.words.extend(words)

    def to_flat(self, word_list):
        return [item for sublist in word_list for item in sublist]

    def get_words(self, flat=False):
        if flat:
            return self.to_flat(self.words)
        return self.words

    def create_inputs_and_labels(self):
        inputs = []
        labels = []

        last_sentence = None
        regx = re.compile('^:[\w_-]+:$')
        for line in self.words:
            reaction_emojis = [w for w in line if regx.match(w)]
            reaction_emojis = list(set(reaction_emojis))  # uniq
            line_wo_emojis = [w for w in line if not regx.match(w)]
            if self.seq_len and len(line_wo_emojis) > self.seq_len:
                continue
            if line_wo_emojis:
                # sentence and reaction line
                for emoji in reaction_emojis:
                    inputs.append(line_wo_emojis)
                    labels.append(emoji)
                last_sentence = line_wo_emojis
            # else:
            #     # reaction only line
            #     for emoji in reaction_emojis:
            #         inputs.append(last_sentence)
            #         labels.append(emoji)
        self.inputs = inputs
        self.labels = labels
        return inputs, labels

    def reaction_mapping(self):
        if not self.reaction_map:
            return self.inputs, self.labels

        with open(self.reaction_map) as f:
            mapping = yaml.load(f)

        mapped_inputs_dict = {}
        mapped_labels_dict = {}

        for l in mapping:
            mapped_inputs_dict[l] = []
            mapped_labels_dict[l] = []

        for (input, label) in zip(self.inputs, self.labels):
            for l in mapping:
                if label in mapping[l]:
                    mapped_inputs_dict[l].append(input)
                    mapped_labels_dict[l].append(l)
                    break

        mapped_inputs = []
        mapped_labels = []

        _len = [len(v) for (k, v) in mapped_labels_dict.items()]
        if self.even:
            label_len = min(_len)
        else:
            label_len = max(_len)

        last_inputs = {}
        for i in range(label_len):
            for l in mapping:
                if i < len(mapped_inputs_dict[l]):
                    mapped_input = mapped_inputs_dict[l][i]
                    mapped_label = mapped_labels_dict[l][i]
                    if l in last_inputs:
                        if last_inputs[l] != mapped_input:
                            mapped_inputs.append(mapped_input)
                            mapped_labels.append(mapped_label)
                    else:
                        mapped_inputs.append(mapped_input)
                        mapped_labels.append(mapped_label)
                    last_inputs[l] = mapped_input
        self.inputs = mapped_inputs
        self.labels = mapped_labels

        return self.inputs, self.labels

    def write_inputs(self, filename):
        punctuations = ["、", "。"]
        with open(filename, 'w') as f:
            for text in self.inputs:
                text_no_punctuation = [x for x in text if x not in punctuations]
                f.write(' '.join(text_no_punctuation) + "\n")

    def write_labels(self, filename):
        with open(filename, 'w') as f:
            f.writelines([x + "\n" for x in self.labels])

    def write_inputs_and_labels(self, filename):
        with open(filename, 'w') as f:
            f.writelines("\n".join([str(x) for x in zip(self.inputs, self.labels)]))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('files', action='store', type=str, nargs='+')
    parser.add_argument('--mapfile', action='store', type=str)
    args = parser.parse_args()

    dataset = SlackMessageDataset(reaction_map=args.mapfile, even=True)
    for filename in args.files:
        dataset.add_file(filename)
    print('Add {0} files'.format(len(args.files)))

    print('Creating dataset...')
    inputs, labels = dataset.create_inputs_and_labels()

    inputs, labels = dataset.reaction_mapping()

    print('Created.')
    dataset.write_inputs("data/slack_comments.txt")
    dataset.write_labels("data/slack_labels.txt")
    dataset.write_inputs_and_labels("data/slack_comments_and_labels.txt")
    print('labels=', Counter(labels))
