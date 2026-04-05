"""
indexer.py
----------
Unit 3 — Indexing and Data Mining Techniques
Covers: Inverted Files, Dictionaries, Encoding, Static & Dynamic Inverted Indices,
        Scalable Indexing, Index Compression, TF-IDF Term Weighting (Unit 2)
"""

import json
import math
import os
import pickle
from collections import defaultdict, Counter

from .preprocessor import Preprocessor


class Indexer:
    """
    Builds and manages an inverted index over the notes.json database.
    Implements TF-IDF weighting, inverted file structure, and index compression.
    """

    def __init__(self, data_path='data/notes.json'):
        self.data_path = data_path
        self.preprocessor = Preprocessor()

        # Core index structures
        self.documents = []               # Raw document list
        self.inverted_index = defaultdict(list)  # term -> [(doc_id, tf, positions)]
        self.tfidf_matrix = {}            # doc_id -> {term: tfidf_weight}
        self.doc_lengths = {}             # doc_id -> vector magnitude (for cosine)
        self.doc_freq = defaultdict(int)  # term -> number of docs containing term
        self.term_freq = {}               # doc_id -> {term: raw_count}
        self.vocabulary = set()           # All unique terms
        self.total_docs = 0

        # Compressed index (gap encoding)
        self.compressed_index = {}

        # Load and build index immediately
        self.load_documents()
        self.build_index()

    # ------------------------------------------------------------------ #
    #  DOCUMENT LOADING                                                    #
    # ------------------------------------------------------------------ #

    def load_documents(self):
        """Loads documents from notes.json into memory."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            self.total_docs = len(self.documents)
            print(f"[Indexer] Loaded {self.total_docs} documents from {self.data_path}")
        except FileNotFoundError:
            print(f"[Indexer] ERROR: {self.data_path} not found.")
            self.documents = []
            self.total_docs = 0

    def get_documents(self):
        return self.documents

    # ------------------------------------------------------------------ #
    #  UNIT 3 — INVERTED INDEX CONSTRUCTION                               #
    # ------------------------------------------------------------------ #

    def build_index(self):
        """
        Builds the inverted index from all documents.
        Inverted Index: dictionary of terms → postings list (doc_id, tf, positions)
        Also computes TF-IDF weights for all term-document pairs.
        """
        if not self.documents:
            return

        print("[Indexer] Building inverted index...")

        # Reset structures
        self.inverted_index = defaultdict(list)
        self.doc_freq = defaultdict(int)
        self.term_freq = {}
        self.vocabulary = set()

        # Step 1: Process each document — build postings
        for doc in self.documents:
            doc_id = doc['id']
            full_text = doc['topic'] + ' ' + doc['content'] + ' ' + doc.get('unit', '')
            tokens = self.preprocessor.preprocess(full_text)

            # Compute term frequency for this document
            tf_counts = Counter(tokens)
            self.term_freq[doc_id] = dict(tf_counts)

            # Track positions for each term
            term_positions = defaultdict(list)
            for pos, token in enumerate(tokens):
                term_positions[token].append(pos)

            # Update inverted index with (doc_id, tf, positions)
            for term, count in tf_counts.items():
                self.inverted_index[term].append({
                    'doc_id': doc_id,
                    'tf': count,
                    'positions': term_positions[term]
                })
                self.doc_freq[term] += 1
                self.vocabulary.add(term)

        # Step 2: Compute TF-IDF weights
        self._compute_tfidf()

        # Step 3: Build compressed index (gap encoding)
        self._build_compressed_index()

        print(f"[Indexer] Index built. Vocabulary size: {len(self.vocabulary)}")

    # ------------------------------------------------------------------ #
    #  UNIT 2 — TF-IDF TERM WEIGHTING                                     #
    # ------------------------------------------------------------------ #

    def _compute_tfidf(self):
        """
        Computes TF-IDF weights for all (term, document) pairs.

        TF (Term Frequency) — normalized: tf(t,d) = count(t,d) / max_count(d)
        IDF (Inverse Document Frequency): idf(t) = log(N / df(t))
        TF-IDF: tfidf(t,d) = tf(t,d) * idf(t)
        """
        self.tfidf_matrix = {}
        self.doc_lengths = {}

        for doc in self.documents:
            doc_id = doc['id']
            tf_counts = self.term_freq.get(doc_id, {})

            if not tf_counts:
                self.tfidf_matrix[doc_id] = {}
                self.doc_lengths[doc_id] = 0
                continue

            max_tf = max(tf_counts.values()) if tf_counts else 1
            tfidf_scores = {}

            for term, raw_tf in tf_counts.items():
                # Normalized TF
                tf = raw_tf / max_tf
                # IDF with +1 smoothing to avoid division by zero
                df = self.doc_freq.get(term, 1)
                idf = math.log((self.total_docs + 1) / (df + 1)) + 1
                tfidf_scores[term] = round(tf * idf, 6)

            self.tfidf_matrix[doc_id] = tfidf_scores

            # Compute document vector magnitude for cosine normalization
            magnitude = math.sqrt(sum(w ** 2 for w in tfidf_scores.values()))
            self.doc_lengths[doc_id] = magnitude if magnitude > 0 else 1.0

    # ------------------------------------------------------------------ #
    #  UNIT 3 — INDEX COMPRESSION (Gap Encoding)                          #
    # ------------------------------------------------------------------ #

    def _build_compressed_index(self):
        """
        Gap (Delta) Encoding: stores gaps between consecutive doc IDs
        instead of absolute IDs, producing smaller numbers that compress better.
        Example: [3, 7, 12, 20] → gaps: [3, 4, 5, 8]
        """
        self.compressed_index = {}
        for term, postings in self.inverted_index.items():
            doc_ids = sorted([p['doc_id'] for p in postings])
            gaps = []
            prev = 0
            for doc_id in doc_ids:
                gaps.append(doc_id - prev)
                prev = doc_id
            self.compressed_index[term] = gaps

    def decode_compressed_postings(self, term):
        """Decodes gap-encoded postings back to absolute doc IDs."""
        if term not in self.compressed_index:
            return []
        gaps = self.compressed_index[term]
        doc_ids = []
        current = 0
        for gap in gaps:
            current += gap
            doc_ids.append(current)
        return doc_ids

    # ------------------------------------------------------------------ #
    #  UNIT 3 — STATIC AND DYNAMIC INDEX OPERATIONS                       #
    # ------------------------------------------------------------------ #

    def add_document(self, new_doc):
        """
        Dynamic Index Update: adds a new document to the existing index.
        Implements incremental update without full rebuild.
        """
        self.documents.append(new_doc)
        self.total_docs += 1
        doc_id = new_doc['id']

        full_text = new_doc['topic'] + ' ' + new_doc['content'] + ' ' + new_doc.get('unit', '')
        tokens = self.preprocessor.preprocess(full_text)
        tf_counts = Counter(tokens)
        self.term_freq[doc_id] = dict(tf_counts)

        term_positions = defaultdict(list)
        for pos, token in enumerate(tokens):
            term_positions[token].append(pos)

        for term, count in tf_counts.items():
            self.inverted_index[term].append({
                'doc_id': doc_id,
                'tf': count,
                'positions': term_positions[term]
            })
            self.doc_freq[term] += 1
            self.vocabulary.add(term)

        # Recompute TF-IDF for the new document only (partial update)
        max_tf = max(tf_counts.values()) if tf_counts else 1
        tfidf_scores = {}
        for term, raw_tf in tf_counts.items():
            tf = raw_tf / max_tf
            df = self.doc_freq.get(term, 1)
            idf = math.log((self.total_docs + 1) / (df + 1)) + 1
            tfidf_scores[term] = round(tf * idf, 6)
        self.tfidf_matrix[doc_id] = tfidf_scores
        magnitude = math.sqrt(sum(w ** 2 for w in tfidf_scores.values()))
        self.doc_lengths[doc_id] = magnitude if magnitude > 0 else 1.0

        print(f"[Indexer] Dynamic update: added document id={doc_id}")

    # ------------------------------------------------------------------ #
    #  LOOKUP METHODS                                                      #
    # ------------------------------------------------------------------ #

    def get_postings(self, term):
        """Returns postings list for a term from the inverted index."""
        tokens = self.preprocessor.preprocess(term)
        if not tokens:
            return []
        stemmed_term = tokens[0]
        return self.inverted_index.get(stemmed_term, [])

    def get_tfidf_vector(self, doc_id):
        """Returns the TF-IDF vector for a given document."""
        return self.tfidf_matrix.get(doc_id, {})

    def get_doc_by_id(self, doc_id):
        """Retrieves a document by its ID."""
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None

    def get_index_stats(self):
        """Returns statistics about the current index."""
        total_postings = sum(len(v) for v in self.inverted_index.values())
        avg_postings = total_postings / len(self.inverted_index) if self.inverted_index else 0
        return {
            'total_documents': self.total_docs,
            'vocabulary_size': len(self.vocabulary),
            'total_postings': total_postings,
            'avg_postings_per_term': round(avg_postings, 2),
            'compressed_index_terms': len(self.compressed_index)
        }

    def save_index(self, path='data/index.pkl'):
        """Serializes the index to disk for persistence."""
        with open(path, 'wb') as f:
            pickle.dump({
                'inverted_index': dict(self.inverted_index),
                'tfidf_matrix': self.tfidf_matrix,
                'doc_lengths': self.doc_lengths,
                'doc_freq': dict(self.doc_freq),
                'vocabulary': self.vocabulary,
                'total_docs': self.total_docs
            }, f)
        print(f"[Indexer] Index saved to {path}")

    def load_index(self, path='data/index.pkl'):
        """Loads a serialized index from disk."""
        if not os.path.exists(path):
            print(f"[Indexer] No saved index found at {path}. Building fresh.")
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.inverted_index = defaultdict(list, data['inverted_index'])
        self.tfidf_matrix = data['tfidf_matrix']
        self.doc_lengths = data['doc_lengths']
        self.doc_freq = defaultdict(int, data['doc_freq'])
        self.vocabulary = data['vocabulary']
        self.total_docs = data['total_docs']
        print(f"[Indexer] Index loaded from {path}")
        return True