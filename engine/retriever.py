"""
retriever.py
------------
Unit 2 — Modeling and Performance Evaluation
Covers: Vector Space Model, Cosine Similarity, Probabilistic Model (BM25),
        Query Processing, Query Refinement, Query Optimization,
        Relevance Feedback (Rocchio Algorithm)
"""

import math
from collections import Counter
from .preprocessor import Preprocessor


class Retriever:
    """
    Core retrieval engine implementing VSM (Vector Space Model) with
    TF-IDF + Cosine Similarity, BM25 probabilistic ranking,
    and Rocchio relevance feedback for query refinement.
    """

    def __init__(self, indexer):
        self.indexer = indexer
        self.preprocessor = Preprocessor()

        # BM25 parameters (tunable)
        self.bm25_k1 = 1.5   # term frequency saturation
        self.bm25_b = 0.75   # document length normalization

        # Compute average document length for BM25
        self._avg_doc_len = self._compute_avg_doc_len()

    def _compute_avg_doc_len(self):
        """Computes average document length (in tokens) for BM25 normalization."""
        if not self.indexer.documents:
            return 1
        total = sum(
            sum(self.indexer.term_freq.get(doc['id'], {}).values())
            for doc in self.indexer.documents
        )
        return total / len(self.indexer.documents)

    # ------------------------------------------------------------------ #
    #  UNIT 2 — QUERY PROCESSING                                           #
    # ------------------------------------------------------------------ #

    def process_query(self, query_string):
        """
        Query Processing Pipeline:
        1. Tokenize query
        2. Remove stopwords
        3. Lemmatize + Stem (same as document preprocessing for consistency)
        4. Compute TF-IDF weights for query terms
        Returns: dict {term: tfidf_weight}
        """
        tokens = self.preprocessor.preprocess(query_string)
        if not tokens:
            return {}

        tf_counts = Counter(tokens)
        max_tf = max(tf_counts.values()) if tf_counts else 1
        query_vector = {}

        for term, count in tf_counts.items():
            tf = count / max_tf
            df = self.indexer.doc_freq.get(term, 0)
            if df == 0:
                continue  # Term not in index — skip
            idf = math.log((self.indexer.total_docs + 1) / (df + 1)) + 1
            query_vector[term] = tf * idf

        return query_vector

    # ------------------------------------------------------------------ #
    #  UNIT 2 — VECTOR SPACE MODEL + COSINE SIMILARITY                    #
    # ------------------------------------------------------------------ #

    def cosine_similarity(self, query_vector, doc_id):
        """
        Computes cosine similarity between query and document vectors.
        cos(θ) = (Q · D) / (|Q| × |D|)
        Returns a score between 0 (no similarity) and 1 (identical).
        """
        doc_vector = self.indexer.tfidf_matrix.get(doc_id, {})
        if not doc_vector or not query_vector:
            return 0.0

        # Dot product: sum of products of matching term weights
        dot_product = sum(
            query_vector.get(term, 0) * doc_vector.get(term, 0)
            for term in query_vector
        )

        # Query magnitude
        query_magnitude = math.sqrt(sum(w ** 2 for w in query_vector.values()))

        # Document magnitude (precomputed in indexer)
        doc_magnitude = self.indexer.doc_lengths.get(doc_id, 1.0)

        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0

        return dot_product / (query_magnitude * doc_magnitude)

    def vsm_retrieve(self, query_string, top_k=10):
        """
        Vector Space Model retrieval:
        Ranks all documents by cosine similarity with the query vector.
        Returns top-k documents sorted by descending similarity score.
        """
        query_vector = self.process_query(query_string)
        if not query_vector:
            return []

        scores = []
        for doc in self.indexer.documents:
            doc_id = doc['id']
            sim = self.cosine_similarity(query_vector, doc_id)
            if sim > 0:
                scores.append((doc_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------ #
    #  UNIT 2 — PROBABILISTIC MODEL (BM25)                                #
    # ------------------------------------------------------------------ #

    def bm25_score(self, query_tokens, doc_id):
        """
        BM25 (Best Match 25) — Probabilistic Relevance Model.
        Improves on TF-IDF with:
        - Term frequency saturation (k1): prevents high-freq terms dominating
        - Document length normalization (b): penalizes very long documents
        BM25(t,d,q) = IDF(t) * [tf(t,d) * (k1+1)] / [tf(t,d) + k1*(1-b+b*(|d|/avgdl))]
        """
        tf_counts = self.indexer.term_freq.get(doc_id, {})
        doc_len = sum(tf_counts.values())
        score = 0.0

        for term in query_tokens:
            if term not in tf_counts:
                continue
            tf = tf_counts[term]
            df = self.indexer.doc_freq.get(term, 0)
            if df == 0:
                continue

            # IDF component
            idf = math.log((self.indexer.total_docs - df + 0.5) / (df + 0.5) + 1)

            # TF component with saturation and length normalization
            tf_norm = (tf * (self.bm25_k1 + 1)) / (
                tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_len / self._avg_doc_len)
            )
            score += idf * tf_norm

        return score

    def bm25_retrieve(self, query_string, top_k=10):
        """
        BM25 retrieval: ranks documents using the BM25 probabilistic model.
        """
        query_tokens = self.preprocessor.preprocess(query_string)
        if not query_tokens:
            return []

        scores = []
        for doc in self.indexer.documents:
            doc_id = doc['id']
            score = self.bm25_score(query_tokens, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------ #
    #  UNIT 2 — QUERY REFINEMENT: ROCCHIO ALGORITHM                       #
    # ------------------------------------------------------------------ #

    def rocchio_expand(self, query_string, relevant_doc_ids,
                       non_relevant_doc_ids=None, alpha=1.0, beta=0.75, gamma=0.15):
        """
        Rocchio Algorithm for Relevance Feedback / Query Expansion.
        Moves the query vector towards relevant documents and away from
        non-relevant documents in vector space.

        Q_new = α*Q_orig + β*(1/|Rel|)*ΣD_rel - γ*(1/|NonRel|)*ΣD_nonrel

        Returns an expanded query token list.
        """
        query_vector = self.process_query(query_string)
        if not query_vector:
            return self.preprocessor.preprocess(query_string)

        # Add centroid of relevant docs
        if relevant_doc_ids:
            rel_centroid = self._compute_centroid(relevant_doc_ids)
            for term, weight in rel_centroid.items():
                query_vector[term] = alpha * query_vector.get(term, 0) + beta * weight

        # Subtract centroid of non-relevant docs
        if non_relevant_doc_ids:
            nonrel_centroid = self._compute_centroid(non_relevant_doc_ids)
            for term, weight in nonrel_centroid.items():
                query_vector[term] = query_vector.get(term, 0) - gamma * weight

        # Remove negative weights and sort by weight
        expanded_terms = sorted(
            [(t, w) for t, w in query_vector.items() if w > 0],
            key=lambda x: x[1], reverse=True
        )
        # Return top-15 terms as expanded query tokens
        return [t for t, _ in expanded_terms[:15]]

    def _compute_centroid(self, doc_ids):
        """Computes the centroid vector of a set of documents."""
        centroid = Counter()
        for doc_id in doc_ids:
            vec = self.indexer.tfidf_matrix.get(doc_id, {})
            centroid.update(vec)
        if doc_ids:
            for term in centroid:
                centroid[term] /= len(doc_ids)
        return dict(centroid)

    # ------------------------------------------------------------------ #
    #  MAIN SEARCH — COMBINES VSM + BM25                                  #
    # ------------------------------------------------------------------ #

    def search(self, query_string, top_k=10, model='vsm'):
        """
        Main search function. Supports VSM (cosine) and BM25 models.
        Returns list of dicts with full document details and scores.
        """
        if model == 'bm25':
            ranked = self.bm25_retrieve(query_string, top_k)
        else:
            ranked = self.vsm_retrieve(query_string, top_k)

        results = []
        for doc_id, score in ranked:
            doc = self.indexer.get_doc_by_id(doc_id)
            if not doc:
                continue

            # Highlight matching terms in content
            query_tokens = self.preprocessor.preprocess(query_string)
            snippet = self._generate_snippet(doc['content'], query_tokens)

            results.append({
                'id': doc_id,
                'topic': doc['topic'],
                'unit': doc['unit'],
                'subject': doc.get('subject', 'IRT'),
                'content': doc['content'],
                'snippet': snippet,
                'score': round(score, 4),
                'model': model.upper()
            })

        return results

    def _generate_snippet(self, content, query_tokens, window=30):
        """
        Generates a relevant snippet from document content by finding
        the sentence with the most query term matches.
        """
        sentences = self.preprocessor.sentence_tokenize(content)
        if not sentences:
            return content[:200] + '...'

        best_sentence = sentences[0]
        best_score = 0
        for sentence in sentences:
            sent_tokens = set(self.preprocessor.preprocess(sentence))
            overlap = len(sent_tokens & set(query_tokens))
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence

        if len(best_sentence) > 250:
            best_sentence = best_sentence[:250] + '...'
        return best_sentence

    def get_query_terms(self, query_string):
        """Returns processed query terms with their weights. Used by UI."""
        qv = self.process_query(query_string)
        return sorted(qv.items(), key=lambda x: x[1], reverse=True)