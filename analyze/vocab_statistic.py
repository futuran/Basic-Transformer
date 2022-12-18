import collections
import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', default='/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec/aspec_train_h100k.en.tkn.bpe')
    parser.add_argument('-t', '--tgt', default='/mnt/work/20221004_RetrieveEditRerank-NMT/data/aspec/aspec_train_h100k.ja.tkn.bpe')

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    vocabs = []
    with open(args.src, 'r') as f:
        for l in f:
            vocabs += l.strip().split()
    with open(args.tgt, 'r') as f:
        for l in f:
            vocabs += l.strip().split()

    vocabs = collections.Counter(vocabs)

    with open('vocab_statistic.csv', 'w') as f:
        for vocab in sorted(vocabs.values(), reverse=True):
            f.write(str(vocab) + '\n')


if __name__ == '__main__':
    main()
