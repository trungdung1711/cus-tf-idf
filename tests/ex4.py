from cus_tf_idf import TFIDF
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def simple_lemmatizer(token: str) -> str:
    if token == "computational":
        return "computer"
    elif token == "algorithms":
        return "algorithm"
    elif token == "structures":
        return "structure"
    else:
        return token


docs = [
    "Data Base System Concepts",
    "Introduction to Algorithms",
    "Computational Geometry Algorithms and Applications",
    "Data Structures and Algorithm Analysis on Massive Data Sets",
    "Computer Organization",
]


def simple_preprocess(doc: str) -> str:
    lower = doc.lower()
    return lower


def simple_tokenizer(doc: str) -> list[str]:
    return doc.split()


vocabs = [
    "data",
    "system",
    "algorithm",
    "computer",
    "geometry",
    "structure",
    "analysis",
    "organization",
]


# preprocess and tokenize doc
def pipeline(docs: list[str]) -> list[list[str]]:
    result = []
    for doc in docs:
        doc = simple_preprocess(doc)
        tokens = simple_tokenizer(doc)
        tokens = [simple_lemmatizer(token) for token in tokens]
        result.append(tokens)

    return result


my_tfidf = TFIDF(docs=pipeline(docs), vocabs=vocabs)
my_tfidf.fit()

new_doc = "Geometry Algorithm Concepts"

doc_tokens = pipeline([new_doc])[0]

query = my_tfidf.transform(doc_tokens)
sims = cosine_similarity(X=np.array([query]), Y=my_tfidf.tf_idf_mat)
sims = sims.flatten()

# Rank documents by similarity (highest first)
ranking = sims.argsort()[::-1]

print("Similarity scores:", sims)
print("Ranking:", ranking)
