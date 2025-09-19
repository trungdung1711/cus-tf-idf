from nltk.tokenize import word_tokenize
from cus_tf_idf import TFIDF
import numpy as np


docs = [
    "Romeo loves Juliet",
    "Juliet loves Shakespeare",
    "Romeo loves Shakespeare"
]

lower_docs = [doc.lower() for doc in docs]

tokenized_docs = [word_tokenize(text=doc) for doc in lower_docs]

tf_idf = TFIDF(docs=tokenized_docs)
result = np.array(tf_idf.tf_idf())
print(result)