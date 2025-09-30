import numpy as np
import math
from collections import Counter


class TFIDF:
    def __init__(self, docs: list[list[str]], vocabs: list[str] = None):
        self.docs = docs
        self.N = len(docs)

        # all tokens including the repeated one
        tokens = [token for doc in self.docs for token in doc]
        # remove repeated one
        self.unsorted_tokens = set(tokens)
        # sort it
        self.tokens = sorted(self.unsorted_tokens)

        if vocabs is not None:
            # defined vocabs, not from the doc
            self.V = len(vocabs)
            self.vocab = {word: index for index, word in enumerate(vocabs)}
        else:
            # vocabs is None, derive the vocabulary
            self.V = len(self.tokens)
            enum = enumerate(self.tokens)
            self.vocab = {word: index for index, word in enum}

    def squash(self, tf: float):
        return 1 + math.log10(tf) if tf > 0 else 0.00

    def tf(self):
        dt_mat = np.zeros(shape=(self.N, self.V), dtype=float)

        for doc_index, doc in enumerate(self.docs):
            # fill in the dt_mat for the current doc
            counter = Counter(doc)
            # word : count
            for word, count in counter.items():
                if word in self.vocab:
                    # get the index
                    word_index = self.vocab[word]
                    dt_mat[doc_index, word_index] = count

                # word not in vocab -> skip

        self.term_frequency = dt_mat

        tf_mat = np.zeros_like(self.term_frequency, dtype=float)
        mask = self.term_frequency > 0
        # squash step, only squash the positive
        tf_mat[mask] = 1 + np.log10(self.term_frequency[mask])
        return tf_mat

    def idf(self):
        self.tf()
        self.document_frequency = (self.term_frequency > 0).sum(axis=0)
        return np.log10(self.N / self.document_frequency)

    def tf_idf(self):
        self.values = self.tf() * self.idf()
        return self.values

    def fit(self):
        # for this training set, we would learn the vocab, representation
        self.tf_mat = self.tf()
        self.idf_mat = self.idf()
        self.tf_idf_mat = self.tf_idf()

    def transform(self, doc: list[str]):
        # based on the current vocabulary
        # tf
        tf_arr = np.zeros(shape=(self.V), dtype=float)
        counter = Counter(doc)
        # word : count
        for word, count in counter.items():
            if word in self.vocab:
                # get the index
                word_index = self.vocab[word]
                tf_arr[word_index] = count

        mask = tf_arr > 0
        tf_arr[mask] = 1 + np.log10(tf_arr[mask])
        return tf_arr * self.idf_mat
