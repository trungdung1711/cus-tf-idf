import numpy as np
import math
from collections import Counter


class TFIDF:
    def __init__(self, docs : list[list[str]]):
        self.docs = docs
        self.N = len(docs)
        
        self.tokens = sorted(set([token for doc in self.docs for token in doc]))
        self.V = len(self.tokens)
        self.vocab = {word : index for index, word in enumerate(self.tokens)}
        
        
    def tf(self):
        dt_mat = np.zeros(shape=(self.N, self.V), dtype=float)
        
        def squash(tf : float):
            return 1 + math.log10(tf) if tf > 0 else 0.00
                
        for doc_index, doc in enumerate(self.docs):
            # fill in the dt_mat for the current doc
            counter = Counter(doc)
            # word : count
            for word, count in counter.items():
                word_index = self.vocab[word]
                dt_mat[doc_index, word_index] = count
        
        self.term_frequency = dt_mat
        
        tf_mat = np.zeros_like(self.term_frequency, dtype=float)
        mask = self.term_frequency > 0
        # squash step
        tf_mat[mask] = 1 + np.log10(self.term_frequency[mask])
        return tf_mat
                

    def idf(self):
        self.tf()
        self.inverse_document_frequency = (self.term_frequency > 0).sum(axis=0)
        return np.log10(self.N / self.inverse_document_frequency)
    

    def tf_idf(self):
        return self.tf() * self.idf()