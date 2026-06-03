# ============================================================
# vectorization.py — Les 3 méthodes de vectorisation
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from gensim.models import Word2Vec


def get_vectorizers():
    """
    Retourne les 3 vectoriseurs :
      - binary : 1 si le mot existe, 0 sinon
      - count  : nombre d'occurrences du mot
      - tfidf  : poids selon l'importance du mot dans le document
    Tous utilisent des bigrams (1,2) pour capturer des expressions
    comme "not good" ou "very happy".
    """
    return {
        "binary": CountVectorizer(
            binary=True,
            ngram_range=(1, 2),
            max_features=50000
        ),
        "count": CountVectorizer(
            binary=False,
            ngram_range=(1, 2),
            max_features=50000
        ),
        "tfidf": TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000
        ),
    }

def get_word2vec_vector(X_text, vector_size=100):
    """
    Vectorise les tweets avec Word2Vec.
    Chaque tweet = moyenne des vecteurs de ses mots.
    """

    # 1. Tokeniser chaque tweet en liste de mots
    sentences = [text.split() for text in X_text]

    # 2. Entraîner le modèle Word2Vec
    w2v_model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        epochs=10
    )

    # 3. Chaque tweet = moyenne des vecteurs de ses mots
    def tweet_vector(text):
        words = text.split()
        vectors = [
            w2v_model.wv[w]
            for w in words
            if w in w2v_model.wv
        ]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(vector_size)

    return np.array([tweet_vector(text) for text in X_text])
