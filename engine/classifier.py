"""
classifier.py
-------------
Unit 3 — Classification Techniques
Covers: K-Nearest Neighbor (KNN), Naive Bayes, Support Vector Machines (SVM)
        Applied to classify notes by unit/topic using TF-IDF vectors
"""

import math
from collections import Counter, defaultdict
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from .preprocessor import Preprocessor


class Classifier:
    """
    Implements KNN, Naive Bayes, and SVM classifiers for note classification.
    Given a text snippet, predicts which IRT unit/topic it belongs to.
    """

    def __init__(self, indexer):
        self.indexer = indexer
        self.preprocessor = Preprocessor()

        # Prepare training data
        self.texts = []
        self.labels_unit = []
        self.labels_topic = []
        self.doc_ids = []

        # sklearn models
        self.nb_pipeline = None
        self.svm_pipeline = None
        self.label_encoder = LabelEncoder()

        # KNN: we use precomputed TF-IDF matrix from the indexer
        self.knn_doc_vectors = {}

        self._prepare_training_data()
        self._train_models()

    # ------------------------------------------------------------------ #
    #  DATA PREPARATION                                                    #
    # ------------------------------------------------------------------ #

    def _prepare_training_data(self):
        """
        Prepares training data from notes.json for classifiers.
        Label = IRT unit (Unit 1 ... Unit 5)
        """
        for doc in self.indexer.documents:
            full_text = doc['topic'] + ' ' + doc['content']
            self.texts.append(full_text)
            self.labels_unit.append(doc['unit'])
            self.labels_topic.append(doc['topic'])
            self.doc_ids.append(doc['id'])

        print(f"[Classifier] Prepared {len(self.texts)} training samples.")

    # ------------------------------------------------------------------ #
    #  UNIT 3 — NAIVE BAYES                                               #
    # ------------------------------------------------------------------ #

    def _train_naive_bayes(self):
        """
        Trains a Multinomial Naive Bayes classifier.
        NB assumes term independence (naive assumption).
        P(unit | text) ∝ P(unit) * Π P(term | unit)
        """
        self.nb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                preprocessor=lambda x: ' '.join(self.preprocessor.preprocess(x)),
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )),
            ('clf', MultinomialNB(alpha=1.0))  # alpha = Laplace smoothing
        ])
        self.nb_pipeline.fit(self.texts, self.labels_unit)
        print("[Classifier] Naive Bayes trained.")

    # ------------------------------------------------------------------ #
    #  UNIT 3 — SUPPORT VECTOR MACHINE                                    #
    # ------------------------------------------------------------------ #

    def _train_svm(self):
        """
        Trains a Linear SVM classifier.
        SVM finds the maximum margin hyperplane in TF-IDF feature space.
        LinearSVC is efficient for high-dimensional text data.
        """
        self.svm_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                preprocessor=lambda x: ' '.join(self.preprocessor.preprocess(x)),
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )),
            ('clf', LinearSVC(C=1.0, max_iter=2000))
        ])
        self.svm_pipeline.fit(self.texts, self.labels_unit)
        print("[Classifier] SVM trained.")

    def _train_models(self):
        """Trains all classifiers."""
        self._train_naive_bayes()
        self._train_svm()

    # ------------------------------------------------------------------ #
    #  UNIT 3 — K-NEAREST NEIGHBOR (Manual Implementation)               #
    # ------------------------------------------------------------------ #

    def knn_classify(self, query_text, k=3):
        """
        K-Nearest Neighbor classification.
        Finds K most similar training documents using cosine similarity
        on TF-IDF vectors. Majority vote determines the class label.
        """
        query_tokens = self.preprocessor.preprocess(query_text)
        query_tf = Counter(query_tokens)
        query_vector = {}
        total_docs = self.indexer.total_docs

        # Build query TF-IDF vector
        for term, count in query_tf.items():
            df = self.indexer.doc_freq.get(term, 0)
            if df == 0:
                continue
            tf = count / max(query_tf.values())
            idf = math.log((total_docs + 1) / (df + 1)) + 1
            query_vector[term] = tf * idf

        if not query_vector:
            return {'label': 'Unknown', 'confidence': 0, 'neighbors': []}

        # Compute cosine similarity with every training doc
        similarities = []
        for doc in self.indexer.documents:
            doc_id = doc['id']
            doc_vector = self.indexer.tfidf_matrix.get(doc_id, {})
            sim = self._cosine(query_vector, doc_vector)
            similarities.append((doc_id, sim, doc['unit'], doc['topic']))

        # Sort by similarity descending and take top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # Majority vote
        unit_votes = Counter(item[2] for item in top_k)
        predicted_unit = unit_votes.most_common(1)[0][0]
        confidence = unit_votes.most_common(1)[0][1] / k

        neighbors = [
            {'doc_id': d, 'similarity': round(s, 4), 'unit': u, 'topic': t}
            for d, s, u, t in top_k
        ]

        return {
            'label': predicted_unit,
            'confidence': round(confidence, 2),
            'neighbors': neighbors,
            'model': 'KNN',
            'k': k
        }

    def _cosine(self, vec_a, vec_b):
        """Computes cosine similarity between two term-weight dicts."""
        dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_a)
        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ------------------------------------------------------------------ #
    #  CLASSIFICATION INTERFACE                                            #
    # ------------------------------------------------------------------ #

    def classify(self, query_text, method='all'):
        """
        Classifies a text snippet using specified method(s).
        Returns predicted unit label and confidence.
        """
        results = {}

        if method in ('all', 'nb'):
            nb_pred = self.nb_pipeline.predict([query_text])[0]
            nb_proba = self.nb_pipeline.predict_proba([query_text])[0]
            nb_conf = round(float(max(nb_proba)), 3)
            results['naive_bayes'] = {'label': nb_pred, 'confidence': nb_conf}

        if method in ('all', 'svm'):
            svm_pred = self.svm_pipeline.predict([query_text])[0]
            results['svm'] = {'label': svm_pred, 'confidence': None}

        if method in ('all', 'knn'):
            knn_result = self.knn_classify(query_text, k=3)
            results['knn'] = knn_result

        # Majority vote across models for final prediction
        if method == 'all':
            votes = Counter()
            votes[results['naive_bayes']['label']] += 1
            votes[results['svm']['label']] += 1
            votes[results['knn']['label']] += 1
            final_label = votes.most_common(1)[0][0]
            results['final_prediction'] = final_label
            results['vote_breakdown'] = dict(votes)

        return results

    def get_top_terms_per_class(self, top_n=5):
        """
        Returns the top N most informative terms for each unit class.
        Extracted from the Naive Bayes feature log probabilities.
        """
        nb_clf = self.nb_pipeline.named_steps['clf']
        tfidf = self.nb_pipeline.named_steps['tfidf']
        feature_names = tfidf.get_feature_names_out()
        class_labels = nb_clf.classes_

        top_terms = {}
        for i, label in enumerate(class_labels):
            log_probs = nb_clf.feature_log_prob_[i]
            top_indices = log_probs.argsort()[-top_n:][::-1]
            top_terms[label] = [feature_names[j] for j in top_indices]

        return top_terms