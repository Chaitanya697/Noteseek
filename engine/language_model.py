"""
language_model.py
-----------------
Unit 2 + Unit 5 — Language Models for IR
Covers: Unigram Language Model, N-gram Models, Query Likelihood,
        Jelinek-Mercer Smoothing, Laplace Smoothing, Document Scoring
"""

import math
from collections import Counter, defaultdict
from .preprocessor import Preprocessor


class LanguageModel:
    """
    Implements Language Model-based Information Retrieval.
    Ranks documents by the probability that the document's language
    model generates the query terms — the Query Likelihood Model.
    """

    def __init__(self, indexer, smoothing='jelinek_mercer', lambda_param=0.7):
        self.indexer = indexer
        self.preprocessor = Preprocessor()
        self.smoothing = smoothing
        self.lambda_param = lambda_param  # for Jelinek-Mercer smoothing

        # Build language model probabilities
        self.doc_lm = {}          # doc_id -> {term: probability}
        self.collection_lm = {}   # term -> collection probability
        self.doc_lengths_lm = {}  # doc_id -> total token count

        self._build_language_models()
        self._build_ngram_models()

    # ------------------------------------------------------------------ #
    #  UNIT 2 + 5 — UNIGRAM LANGUAGE MODEL                                #
    # ------------------------------------------------------------------ #

    def _build_language_models(self):
        """
        Builds a unigram language model for each document.
        P(term | doc) = count(term, doc) / total_terms_in_doc
        Also builds a collection-level language model for smoothing.
        """
        collection_counts = Counter()
        total_collection_tokens = 0

        for doc in self.indexer.documents:
            doc_id = doc['id']
            tf_counts = self.indexer.term_freq.get(doc_id, {})
            total_tokens = sum(tf_counts.values())
            self.doc_lengths_lm[doc_id] = max(total_tokens, 1)

            # Unigram probability for each term in document
            doc_prob = {}
            for term, count in tf_counts.items():
                doc_prob[term] = count / max(total_tokens, 1)
            self.doc_lm[doc_id] = doc_prob

            # Accumulate collection counts
            collection_counts.update(tf_counts)
            total_collection_tokens += total_tokens

        # Collection-level language model
        total_collection_tokens = max(total_collection_tokens, 1)
        for term, count in collection_counts.items():
            self.collection_lm[term] = count / total_collection_tokens

        print(f"[LanguageModel] Built unigram LMs for {len(self.doc_lm)} documents.")

    # ------------------------------------------------------------------ #
    #  UNIT 5 — N-GRAM LANGUAGE MODELS                                    #
    # ------------------------------------------------------------------ #

    def _build_ngram_models(self):
        """
        Builds bigram language models for each document.
        P(w2 | w1) = count(w1, w2) / count(w1)
        """
        self.bigram_lm = {}  # doc_id -> {(w1,w2): probability}

        for doc in self.indexer.documents:
            doc_id = doc['id']
            full_text = doc['topic'] + ' ' + doc['content']
            tokens = self.preprocessor.preprocess(full_text)

            bigram_counts = Counter()
            unigram_counts = Counter(tokens)

            for i in range(len(tokens) - 1):
                bigram_counts[(tokens[i], tokens[i + 1])] += 1

            bigram_probs = {}
            for (w1, w2), count in bigram_counts.items():
                bigram_probs[(w1, w2)] = count / max(unigram_counts[w1], 1)

            self.bigram_lm[doc_id] = bigram_probs

    # ------------------------------------------------------------------ #
    #  SMOOTHING METHODS                                                   #
    # ------------------------------------------------------------------ #

    def _jelinek_mercer_smooth(self, term, doc_id):
        """
        Jelinek-Mercer Smoothing:
        P_smooth(t|d) = λ * P(t|d) + (1-λ) * P(t|collection)
        λ close to 1 trusts the document more.
        λ close to 0 trusts the collection more (better for short queries).
        """
        p_term_doc = self.doc_lm.get(doc_id, {}).get(term, 0)
        p_term_col = self.collection_lm.get(term, 1e-10)
        return self.lambda_param * p_term_doc + (1 - self.lambda_param) * p_term_col

    def _laplace_smooth(self, term, doc_id):
        """
        Laplace (Add-1) Smoothing:
        P_smooth(t|d) = (count(t,d) + 1) / (total_tokens_in_d + |vocab|)
        Ensures non-zero probability for every term.
        """
        vocab_size = len(self.indexer.vocabulary)
        tf_counts = self.indexer.term_freq.get(doc_id, {})
        count_t_d = tf_counts.get(term, 0)
        total_d = self.doc_lengths_lm.get(doc_id, 1)
        return (count_t_d + 1) / (total_d + vocab_size)

    def _dirichlet_smooth(self, term, doc_id, mu=2000):
        """
        Dirichlet Smoothing:
        P_smooth(t|d) = (count(t,d) + mu * P(t|col)) / (total_d + mu)
        mu controls how much to trust the collection model.
        """
        tf_counts = self.indexer.term_freq.get(doc_id, {})
        count_t_d = tf_counts.get(term, 0)
        total_d = self.doc_lengths_lm.get(doc_id, 1)
        p_term_col = self.collection_lm.get(term, 1e-10)
        return (count_t_d + mu * p_term_col) / (total_d + mu)

    def _get_smoothed_prob(self, term, doc_id):
        """Dispatches to the selected smoothing method."""
        if self.smoothing == 'jelinek_mercer':
            return self._jelinek_mercer_smooth(term, doc_id)
        elif self.smoothing == 'laplace':
            return self._laplace_smooth(term, doc_id)
        elif self.smoothing == 'dirichlet':
            return self._dirichlet_smooth(term, doc_id)
        else:
            return self._jelinek_mercer_smooth(term, doc_id)

    # ------------------------------------------------------------------ #
    #  DOCUMENT SCORING — QUERY LIKELIHOOD                                 #
    # ------------------------------------------------------------------ #

    def score_document(self, query_tokens, doc_id):
        """
        Query Likelihood Score: log P(query | doc_LM)
        = sum of log P(t | doc) for each query term t.
        Uses log probabilities to avoid underflow with long queries.
        """
        log_prob = 0.0
        for term in query_tokens:
            p = self._get_smoothed_prob(term, doc_id)
            log_prob += math.log(max(p, 1e-10))
        return log_prob

    def retrieve(self, query_string, top_k=10):
        """
        Retrieves top-k documents ranked by query likelihood.
        Returns list of (doc_id, score) tuples sorted by descending score.
        """
        query_tokens = self.preprocessor.preprocess(query_string)
        if not query_tokens:
            return []

        scores = []
        for doc in self.indexer.documents:
            doc_id = doc['id']
            score = self.score_document(query_tokens, doc_id)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def retrieve_with_docs(self, query_string, top_k=10):
        """
        Retrieves top-k documents with full document details.
        """
        ranked = self.retrieve(query_string, top_k)
        results = []
        for doc_id, score in ranked:
            doc = self.indexer.get_doc_by_id(doc_id)
            if doc:
                # Normalize score to 0-1 range for display
                norm_score = round(1 / (1 + abs(score)), 4)
                results.append({
                    'id': doc_id,
                    'topic': doc['topic'],
                    'unit': doc['unit'],
                    'content': doc['content'][:250] + '...' if len(doc['content']) > 250 else doc['content'],
                    'score': norm_score,
                    'raw_score': round(score, 4),
                    'model': f'Language Model ({self.smoothing})'
                })
        return results

    def get_ngram_score(self, query_string, doc_id, n=2):
        """
        Scores a document using n-gram language model probabilities.
        Captures local word order beyond bag-of-words.
        """
        tokens = self.preprocessor.preprocess(query_string)
        if len(tokens) < n:
            return self.score_document(tokens, doc_id)

        bigram_lm = self.bigram_lm.get(doc_id, {})
        log_prob = 0.0
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            p = bigram_lm.get(bigram, 1e-6)
            log_prob += math.log(p)
        return log_prob