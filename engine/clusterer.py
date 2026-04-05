"""
clusterer.py
------------
Unit 3 — Clustering Techniques
Covers: K-Means Clustering (Partitioning Method),
        Hierarchical Clustering (Agglomerative), Dendrogram
"""

import math
import random
from collections import defaultdict, Counter

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

from .preprocessor import Preprocessor


class Clusterer:
    """
    Clusters the notes collection using K-Means and Hierarchical methods.
    Useful for organizing notes by topic similarity and discovering themes.
    """

    def __init__(self, indexer):
        self.indexer = indexer
        self.preprocessor = Preprocessor()

        # Vectorize all documents with TF-IDF for sklearn clustering
        self.tfidf_vectorizer = TfidfVectorizer(
            preprocessor=lambda x: ' '.join(self.preprocessor.preprocess(x)),
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True,
            min_df=1
        )

        self.doc_texts = []
        self.doc_ids = []
        self.doc_topics = []
        self.doc_units = []
        self.X = None  # TF-IDF matrix

        self._prepare_data()

    def _prepare_data(self):
        """Prepares TF-IDF feature matrix for all documents."""
        for doc in self.indexer.documents:
            self.doc_texts.append(doc['topic'] + ' ' + doc['content'])
            self.doc_ids.append(doc['id'])
            self.doc_topics.append(doc['topic'])
            self.doc_units.append(doc['unit'])

        if self.doc_texts:
            self.X = self.tfidf_vectorizer.fit_transform(self.doc_texts)
            print(f"[Clusterer] TF-IDF matrix shape: {self.X.shape}")

    # ------------------------------------------------------------------ #
    #  UNIT 3 — K-MEANS CLUSTERING (sklearn)                              #
    # ------------------------------------------------------------------ #

    def kmeans_cluster(self, k=5, random_state=42):
        """
        K-Means Clustering:
        1. Initialize K centroids randomly
        2. Assign each document to nearest centroid (by cosine/euclidean)
        3. Recompute centroids as cluster means
        4. Repeat until convergence

        Returns cluster assignments with top terms per cluster.
        """
        if self.X is None or self.X.shape[0] == 0:
            return {'error': 'No documents to cluster'}

        n_samples = self.X.shape[0]
        k = min(k, n_samples)

        km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                    max_iter=300, random_state=random_state)
        labels = km.fit_predict(self.X)

        # Compute silhouette score (quality measure)
        if n_samples > k:
            try:
                sil_score = round(silhouette_score(self.X, labels, metric='cosine'), 4)
            except Exception:
                sil_score = None
        else:
            sil_score = None

        # Group documents by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[int(label)].append({
                'id': self.doc_ids[idx],
                'topic': self.doc_topics[idx],
                'unit': self.doc_units[idx]
            })

        # Get top terms for each cluster centroid
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        cluster_top_terms = {}
        for cluster_id in range(k):
            centroid = km.cluster_centers_[cluster_id]
            top_indices = centroid.argsort()[-8:][::-1]
            cluster_top_terms[cluster_id] = [feature_names[i] for i in top_indices]

        return {
            'method': 'K-Means',
            'k': k,
            'silhouette_score': sil_score,
            'clusters': [
                {
                    'cluster_id': cid,
                    'size': len(docs),
                    'top_terms': cluster_top_terms.get(cid, []),
                    'documents': docs
                }
                for cid, docs in sorted(clusters.items())
            ]
        }

    # ------------------------------------------------------------------ #
    #  UNIT 3 — K-MEANS MANUAL IMPLEMENTATION                             #
    # ------------------------------------------------------------------ #

    def kmeans_manual(self, k=5, max_iter=100):
        """
        Manual K-Means implementation using cosine distance on
        the indexer's TF-IDF vectors. Shows the algorithm clearly.
        """
        docs = self.indexer.documents
        if not docs or k > len(docs):
            return {}

        doc_vectors = {
            doc['id']: self.indexer.tfidf_matrix.get(doc['id'], {})
            for doc in docs
        }

        # Step 1: Random initialization of K centroids
        all_ids = list(doc_vectors.keys())
        random.seed(42)
        centroid_ids = random.sample(all_ids, k)
        centroids = {i: dict(doc_vectors[cid]) for i, cid in enumerate(centroid_ids)}

        assignments = {}
        for iteration in range(max_iter):
            new_assignments = {}

            # Step 2: Assign each doc to nearest centroid
            for doc_id, vec in doc_vectors.items():
                best_cluster = 0
                best_sim = -1
                for cid, centroid in centroids.items():
                    sim = self._cosine(vec, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster = cid
                new_assignments[doc_id] = best_cluster

            # Step 3: Check convergence
            if new_assignments == assignments:
                print(f"[Clusterer] K-Means converged at iteration {iteration+1}")
                break
            assignments = new_assignments

            # Step 4: Recompute centroids
            cluster_docs = defaultdict(list)
            for doc_id, cluster in assignments.items():
                cluster_docs[cluster].append(doc_vectors[doc_id])

            for cluster_id, vecs in cluster_docs.items():
                if not vecs:
                    continue
                new_centroid = Counter()
                for vec in vecs:
                    new_centroid.update(vec)
                for term in new_centroid:
                    new_centroid[term] /= len(vecs)
                centroids[cluster_id] = dict(new_centroid)

        # Build result
        cluster_groups = defaultdict(list)
        for doc_id, cluster in assignments.items():
            doc = self.indexer.get_doc_by_id(doc_id)
            if doc:
                cluster_groups[cluster].append({
                    'id': doc_id,
                    'topic': doc['topic'],
                    'unit': doc['unit']
                })

        return {
            'method': 'K-Means (Manual)',
            'k': k,
            'iterations': iteration + 1,
            'clusters': [
                {'cluster_id': cid, 'size': len(docs), 'documents': docs}
                for cid, docs in sorted(cluster_groups.items())
            ]
        }

    def _cosine(self, vec_a, vec_b):
        """Cosine similarity between two term-weight dicts."""
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_a)
        mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ------------------------------------------------------------------ #
    #  UNIT 3 — HIERARCHICAL CLUSTERING (Agglomerative)                   #
    # ------------------------------------------------------------------ #

    def hierarchical_cluster(self, n_clusters=5, linkage_method='ward'):
        """
        Agglomerative Hierarchical Clustering:
        - Starts with each document as its own cluster
        - Iteratively merges the two closest clusters
        - Stops when n_clusters remain
        Linkage: 'ward' minimizes within-cluster variance (best for text)
        """
        if self.X is None or self.X.shape[0] == 0:
            return {'error': 'No documents to cluster'}

        n_samples = self.X.shape[0]
        n_clusters = min(n_clusters, n_samples)

        # sklearn agglomerative clustering
        try:
            agg = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method,
                metric='euclidean' if linkage_method == 'ward' else 'cosine'
            )
            labels = agg.fit_predict(self.X.toarray())
        except Exception as e:
            # Fallback to euclidean if cosine fails
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = agg.fit_predict(self.X.toarray())

        # Group documents by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[int(label)].append({
                'id': self.doc_ids[idx],
                'topic': self.doc_topics[idx],
                'unit': self.doc_units[idx]
            })

        # Dominant unit per cluster
        cluster_summary = []
        for cid, docs in sorted(clusters.items()):
            unit_counts = Counter(d['unit'] for d in docs)
            dominant_unit = unit_counts.most_common(1)[0][0]
            cluster_summary.append({
                'cluster_id': cid,
                'size': len(docs),
                'dominant_unit': dominant_unit,
                'unit_distribution': dict(unit_counts),
                'documents': docs
            })

        return {
            'method': f'Hierarchical ({linkage_method} linkage)',
            'n_clusters': n_clusters,
            'linkage': linkage_method,
            'clusters': cluster_summary
        }

    def find_similar_documents(self, doc_id, top_n=5):
        """
        Finds the top-N most similar documents to a given document
        using cosine similarity on TF-IDF vectors.
        Used as a helper for the recommender.
        """
        if doc_id not in [d for d in self.doc_ids]:
            return []
        try:
            idx = self.doc_ids.index(doc_id)
        except ValueError:
            return []

        doc_vec = self.X[idx]
        similarities = []
        for i, other_id in enumerate(self.doc_ids):
            if other_id == doc_id:
                continue
            other_vec = self.X[i]
            # Cosine similarity via dot product of normalized vectors
            dot = (doc_vec * other_vec.T).toarray()[0][0]
            similarities.append((other_id, dot))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]