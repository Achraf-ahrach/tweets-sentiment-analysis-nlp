# ============================================================
# preprocessing.py — The 6 preprocessing methods
# ============================================================

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer    = PorterStemmer()


# ── Common base cleaning for all methods ───────────────────

def clean_base(text):
    """Common cleaning: lowercase, URLs, mentions, punctuation."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)              # remove mentions
    text = re.sub(r'#', '', text)                 # keep the hashtag word text
    text = re.sub(r'[^a-z\s]', '', text)          # keep letters only
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── 6 preprocessing methods ─────────────────────────────────

def preprocess_tokenization(text):
    """Simple tokenization — no word transformations."""
    text = clean_base(text)
    return " ".join([w for w in text.split() if len(w) > 1])


def preprocess_stopwords(text):
    """Tokenization + stop-words removal."""
    text = clean_base(text)
    return " ".join([w for w in text.split()
                     if w not in stop_words and len(w) > 1])


def preprocess_stemming(text):
    """Stop-words + stemming (cats → cat)."""
    text = clean_base(text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join([stemmer.stem(w) for w in words])


def preprocess_lemmatization(text):
    """Stop-words + lemmatization (better → good)."""
    text = clean_base(text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join([lemmatizer.lemmatize(w) for w in words])


def preprocess_stemming_plus(text):
    """Stemming + repeated letters correction (looove → love)."""
    text = clean_base(text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join([stemmer.stem(w) for w in words])


def preprocess_lemma_misspell(text):
    """Lemmatization + repeated letters correction."""
    text = clean_base(text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join([lemmatizer.lemmatize(w) for w in words])


# ── Dictionary of all methods ──────────────────────────────

PREPROCESSING_METHODS = {
    "tokenization":           preprocess_tokenization,
    "stopwords":              preprocess_stopwords,
    "stemming":               preprocess_stemming,
    "lemmatization":          preprocess_lemmatization,
    "stemming_plus":          preprocess_stemming_plus,
    "lemmatization_misspell": preprocess_lemma_misspell,
}