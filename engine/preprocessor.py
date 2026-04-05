"""
preprocessor.py
---------------
Unit 5 - Text Retrieval & NLP Techniques
Covers: Tokenization, Stopword Removal, Stemming, Lemmatization,
        N-gram Models, Token Recognition, Text Cleaning
"""

import re
import string
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

# Download required NLTK resources (safe to call multiple times)
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                 'omw-1.4', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


class Preprocessor:
    """
    Central NLP preprocessing pipeline for NoteSeek.
    Every technique in Unit 5 of the IRT syllabus is implemented here.
    """

    def __init__(self, use_stemming=True, use_lemmatization=True,
                 remove_stopwords=True, ngram_range=(1, 2)):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.ngram_range = ngram_range

        # Stemmers
        self.porter_stemmer = PorterStemmer()
        self.snowball_stemmer = SnowballStemmer("english")

        # Lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Stopwords (Unit 5: stopword removal)
        self.stop_words = set(stopwords.words('english'))
        # Add domain-specific stopwords for IR notes
        extra_stops = {'also', 'may', 'well', 'one', 'two', 'three',
                       'many', 'much', 'often', 'without', 'within',
                       'among', 'however', 'therefore', 'thus'}
        self.stop_words.update(extra_stops)

    # ------------------------------------------------------------------ #
    #  UNIT 5 — TOKENIZATION                                              #
    # ------------------------------------------------------------------ #

    def tokenize(self, text):
        """
        Word tokenization using NLTK's Punkt tokenizer.
        Handles contractions, punctuation, and whitespace.
        """
        if not text:
            return []
        tokens = word_tokenize(text.lower())
        return tokens

    def sentence_tokenize(self, text):
        """
        Sentence tokenization — splits a paragraph into individual sentences.
        Used for passage-level retrieval.
        """
        if not text:
            return []
        return sent_tokenize(text)

    def token_recognition(self, text):
        """
        Token Recognition (Unit 5): identifies meaningful tokens,
        filters out pure punctuation and numbers.
        """
        tokens = self.tokenize(text)
        recognized = []
        for token in tokens:
            # Keep only alphabetic tokens of length >= 2
            if token.isalpha() and len(token) >= 2:
                recognized.append(token)
        return recognized

    # ------------------------------------------------------------------ #
    #  UNIT 5 — STOPWORD REMOVAL                                          #
    # ------------------------------------------------------------------ #

    def remove_stop_words(self, tokens):
        """
        Removes common stopwords that carry low retrieval value.
        Uses NLTK English stopwords + custom additions.
        """
        if not self.remove_stopwords:
            return tokens
        return [t for t in tokens if t not in self.stop_words]

    # ------------------------------------------------------------------ #
    #  UNIT 5 — STEMMING (Porter Stemmer)                                 #
    # ------------------------------------------------------------------ #

    def stem(self, tokens, method='porter'):
        """
        Stemming reduces words to their base/root form using suffix stripping.
        - Porter Stemmer: most widely used English stemmer.
        - Snowball Stemmer: improved version of Porter.
        """
        if not self.use_stemming:
            return tokens
        if method == 'snowball':
            return [self.snowball_stemmer.stem(t) for t in tokens]
        return [self.porter_stemmer.stem(t) for t in tokens]

    # ------------------------------------------------------------------ #
    #  UNIT 5 — LEMMATIZATION (WordNet)                                   #
    # ------------------------------------------------------------------ #

    def lemmatize(self, tokens):
        """
        Lemmatization reduces tokens to their dictionary base form (lemma).
        Uses WordNet to find the correct lemma based on POS context.
        More accurate than stemming — 'better' → 'good', 'mice' → 'mouse'.
        """
        if not self.use_lemmatization:
            return tokens
        try:
            pos_tags = nltk.pos_tag(tokens)
            lemmatized = []
            for token, pos in pos_tags:
                wn_pos = self._get_wordnet_pos(pos)
                lemma = self.lemmatizer.lemmatize(token, pos=wn_pos)
                lemmatized.append(lemma)
            return lemmatized
        except Exception:
            # Fallback: lemmatize without POS
            return [self.lemmatizer.lemmatize(t) for t in tokens]

    def _get_wordnet_pos(self, treebank_tag):
        """Maps Penn Treebank POS tags to WordNet POS tags."""
        from nltk.corpus import wordnet
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    # ------------------------------------------------------------------ #
    #  UNIT 5 — N-GRAM MODELS                                             #
    # ------------------------------------------------------------------ #

    def generate_ngrams(self, tokens, n):
        """
        Generates n-grams from a list of tokens.
        Unigrams (n=1), Bigrams (n=2), Trigrams (n=3).
        N-gram models capture local word context and phrase patterns.
        """
        return list(ngrams(tokens, n))

    def get_ngram_features(self, tokens):
        """
        Returns n-gram features for a token list based on ngram_range.
        Combines unigrams and bigrams by default for richer representation.
        """
        features = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if len(tokens) >= n:
                grams = self.generate_ngrams(tokens, n)
                features.extend([' '.join(g) for g in grams])
        return features

    def ngram_frequency(self, text, n=2):
        """
        Computes frequency distribution of n-grams in a text.
        Useful for language model estimation.
        """
        tokens = self.preprocess(text)
        gram_list = self.generate_ngrams(tokens, n)
        return Counter(gram_list)

    # ------------------------------------------------------------------ #
    #  FULL PIPELINE                                                       #
    # ------------------------------------------------------------------ #

    def preprocess(self, text, method='lemmatize'):
        """
        Full preprocessing pipeline:
        1. Tokenize
        2. Token Recognition (filter punctuation/numbers)
        3. Stopword Removal
        4. Lemmatization OR Stemming
        Returns a list of clean, normalized tokens.
        """
        if not text:
            return []

        # Step 1: Tokenize
        tokens = self.token_recognition(text)

        # Step 2: Remove stopwords
        tokens = self.remove_stop_words(tokens)

        # Step 3: Normalize — lemmatize first, then optionally stem
        if method == 'lemmatize' and self.use_lemmatization:
            tokens = self.lemmatize(tokens)
        if self.use_stemming:
            tokens = self.stem(tokens)

        return tokens

    def preprocess_text_only(self, text):
        """
        Light preprocessing: lowercase + tokenize only (no stemming/lemma).
        Used for display snippets and Boolean matching.
        """
        tokens = self.tokenize(text)
        tokens = [t for t in tokens if t.isalpha() and len(t) >= 2]
        tokens = self.remove_stop_words(tokens)
        return tokens

    def clean_text(self, text):
        """
        Cleans raw text: removes special characters, extra spaces, URLs.
        """
        if not text:
            return ''
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove non-alphanumeric characters (keep spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def get_vocabulary(self, documents):
        """
        Builds vocabulary from a list of raw text documents.
        Returns a sorted list of unique terms.
        """
        vocab = set()
        for doc in documents:
            tokens = self.preprocess(doc)
            vocab.update(tokens)
        return sorted(vocab)

    def compute_term_stats(self, text):
        """
        Returns basic term statistics: token count, unique terms,
        top 10 frequent terms, and bigrams.
        Useful for IR analysis and debugging.
        """
        tokens = self.preprocess(text)
        freq = Counter(tokens)
        bigrams = self.ngram_frequency(text, n=2)
        return {
            'token_count': len(tokens),
            'unique_terms': len(freq),
            'top_terms': freq.most_common(10),
            'top_bigrams': bigrams.most_common(5)
        }