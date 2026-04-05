"""
recommender.py
--------------
Unit 5 — Recommender Systems
Covers: Content-Based Filtering, Collaborative Filtering (simulated),
        Document Similarity, Recommender System Functions
"""

import math
from collections import Counter, defaultdict
from .preprocessor import Preprocessor


class Recommender:
    """
    Recommends related notes based on content similarity and simulated
    user interaction history. Implements both content-based and
    collaborative filtering approaches as described in Unit 5.
    """

    def __init__(self, indexer):
        self.indexer = indexer
        self.preprocessor = Preprocessor()

        # Simulated user interaction log: {user_id: [doc_id, ...]}
        self.user_history = defaultdict(list)
        # Item-item similarity cache
        self._similarity_cache = {}
        # Precompute document similarities
        self._build_similarity_matrix()

    # ------------------------------------------------------------------ #
    #  UNIT 5 — CONTENT-BASED FILTERING                                   #
    # ------------------------------------------------------------------ #

    def _build_similarity_matrix(self):
        """
        Precomputes pairwise cosine similarity between all documents.
        Used for content-based and item-based collaborative filtering.
        """
        docs = self.indexer.documents
        for i, doc_a in enumerate(docs):
            for j, doc_b in enumerate(docs):
                if i >= j:
                    continue
                id_a = doc_a['id']
                id_b = doc_b['id']
                sim = self._cosine_similarity(id_a, id_b)
                self._similarity_cache[(id_a, id_b)] = sim
                self._similarity_cache[(id_b, id_a)] = sim

    def _cosine_similarity(self, id_a, id_b):
        """Cosine similarity between two documents using TF-IDF vectors."""
        vec_a = self.indexer.tfidf_matrix.get(id_a, {})
        vec_b = self.indexer.tfidf_matrix.get(id_b, {})
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_a)
        mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def get_similar_docs(self, doc_id, top_n=5):
        """
        Content-Based Filtering:
        Recommends documents most similar to the given document.
        Similarity is based on TF-IDF cosine similarity of content.
        """
        similarities = []
        for doc in self.indexer.documents:
            other_id = doc['id']
            if other_id == doc_id:
                continue
            key = (doc_id, other_id)
            sim = self._similarity_cache.get(key, self._cosine_similarity(doc_id, other_id))
            similarities.append((other_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top = similarities[:top_n]

        results = []
        for other_id, sim in top:
            doc = self.indexer.get_doc_by_id(other_id)
            if doc:
                results.append({
                    'id': other_id,
                    'topic': doc['topic'],
                    'unit': doc['unit'],
                    'similarity': round(sim, 4),
                    'reason': 'Content similarity (TF-IDF cosine)'
                })
        return results

    def recommend_by_query(self, query_string, top_n=5):
        """
        Recommends related notes for a search query.
        Finds the best matching document first, then recommends similar ones.
        """
        query_tokens = self.preprocessor.preprocess(query_string)
        if not query_tokens:
            return []

        # Score all documents
        scores = []
        for doc in self.indexer.documents:
            doc_id = doc['id']
            doc_vec = self.indexer.tfidf_matrix.get(doc_id, {})
            score = sum(doc_vec.get(t, 0) for t in query_tokens)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        if not scores:
            return []

        # Get top result and find its similar docs
        top_doc_id = scores[0][0]
        return self.get_similar_docs(top_doc_id, top_n)

    # ------------------------------------------------------------------ #
    #  UNIT 5 — COLLABORATIVE FILTERING (SIMULATED)                       #
    # ------------------------------------------------------------------ #

    def log_interaction(self, user_id, doc_id):
        """
        Records a user interaction (view, click) with a document.
        Builds the implicit feedback matrix for collaborative filtering.
        """
        if doc_id not in self.user_history[user_id]:
            self.user_history[user_id].append(doc_id)

    def collaborative_recommend(self, user_id, top_n=5):
        """
        Item-Based Collaborative Filtering:
        1. Find documents the user has interacted with
        2. For each interacted document, find similar documents
        3. Rank candidates by cumulative similarity score
        4. Return top-N unseen documents

        Simulates collaborative filtering without needing multiple users.
        """
        user_docs = self.user_history.get(user_id, [])
        if not user_docs:
            return self._popular_docs(top_n)

        candidate_scores = Counter()
        for doc_id in user_docs:
            similar = self.get_similar_docs(doc_id, top_n=10)
            for rec in similar:
                if rec['id'] not in user_docs:
                    candidate_scores[rec['id']] += rec['similarity']

        # Return top-N unseen candidates
        results = []
        for doc_id, score in candidate_scores.most_common(top_n):
            doc = self.indexer.get_doc_by_id(doc_id)
            if doc:
                results.append({
                    'id': doc_id,
                    'topic': doc['topic'],
                    'unit': doc['unit'],
                    'score': round(score, 4),
                    'reason': 'Collaborative filtering (item-item)'
                })
        return results

    def _popular_docs(self, top_n=5):
        """Returns top-N documents when no user history is available (cold start)."""
        # Fallback: recommend documents from each unit
        seen_units = set()
        results = []
        for doc in self.indexer.documents:
            if doc['unit'] not in seen_units:
                seen_units.add(doc['unit'])
                results.append({
                    'id': doc['id'],
                    'topic': doc['topic'],
                    'unit': doc['unit'],
                    'score': 1.0,
                    'reason': 'Popular (cold start)'
                })
            if len(results) >= top_n:
                break
        return results

    # ------------------------------------------------------------------ #
    #  UNIT 5 — HYBRID RECOMMENDATION                                     #
    # ------------------------------------------------------------------ #

    def hybrid_recommend(self, query_string, user_id='default', top_n=5,
                          content_weight=0.6, collab_weight=0.4):
        """
        Hybrid Recommender: combines content-based and collaborative filtering.
        Final Score = content_weight * content_score + collab_weight * collab_score
        """
        content_recs = self.recommend_by_query(query_string, top_n * 2)
        collab_recs = self.collaborative_recommend(user_id, top_n * 2)

        # Merge scores
        combined = Counter()
        for rec in content_recs:
            combined[rec['id']] += content_weight * rec.get('similarity', 0.5)
        for rec in collab_recs:
            combined[rec['id']] += collab_weight * rec.get('score', 0.5)

        # Build final recommendation list
        results = []
        for doc_id, score in combined.most_common(top_n):
            doc = self.indexer.get_doc_by_id(doc_id)
            if doc:
                results.append({
                    'id': doc_id,
                    'topic': doc['topic'],
                    'unit': doc['unit'],
                    'score': round(score, 4),
                    'reason': 'Hybrid (content + collaborative)'
                })
        return results

    def get_unit_recommendations(self, unit_label, top_n=5):
        """
        Returns recommended notes from the same unit.
        Useful for 'study more about this unit' feature.
        """
        unit_docs = [doc for doc in self.indexer.documents
                     if doc['unit'] == unit_label]
        results = []
        for doc in unit_docs[:top_n]:
            results.append({
                'id': doc['id'],
                'topic': doc['topic'],
                'unit': doc['unit'],
                'score': 1.0,
                'reason': f'Same unit: {unit_label}'
            })
        return results