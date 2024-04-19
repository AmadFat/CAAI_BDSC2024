import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


def get_dataset():
    labeled_train = pd.read_csv("./database/bag_of_words/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("./database/bag_of_words/testData.tsv", header=0, delimiter="\t", quoting=3)
    return labeled_train, test


def get_cleaned_reviews(raw_reviews):
    reviews_html_reduced = [BeautifulSoup(raw_reviews.review[idx], 'html.parser') for idx in range(len(raw_reviews))]
    reviews_letters_only = [re.sub("[^a-zA-Z]", " ", review.get_text()) for review in reviews_html_reduced]
    reviews_lower_only = [review.lower() for review in reviews_letters_only]
    reviews_splited = [review.split() for review in reviews_lower_only]
    search_dict = dict({word: 0 for word in stopwords.words("english")})
    reviews_cleaned = [[w for w in review if w not in search_dict] for review in reviews_splited]
    return reviews_cleaned


class Vocab:
    def __init__(self, tokens: list = None, min_freq: int = 0, reserved_tokens: list = None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = self.count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
        self.token_freqs = [{token: freq} for token, freq in self.token_freqs]

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def count_corpus(self, tokens: list):
        dictionary = {}
        for token in tokens:
            if isinstance(token, str):
                if token in dictionary:
                    dictionary[token] += 1
                else:
                    dictionary[token] = 1
            elif isinstance(token, list):
                for k, v in self.count_corpus(token).items():
                    if k in dictionary:
                        dictionary[k] += v
                    else:
                        dictionary[k] = v
        return dictionary


def get_idx_with_voc(voc, tokens):
    result = [1] * len(tokens)
    for idx, token in enumerate(tokens):
        if isinstance(token, str):
            if token in voc.token_to_idx:
                result[idx] = voc.token_to_idx[token]
            else:
                result[idx] = 0
        elif isinstance(token, list):
            result[idx] = get_idx_with_voc(voc, token)
    return result