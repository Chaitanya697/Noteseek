"""
boolean_model.py
----------------
Unit 2 — IR Models: Boolean Model
Covers: Boolean Query Parsing, AND / OR / NOT operations,
        Postings list merging, Boolean retrieval over inverted index
"""

import re
from .preprocessor import Preprocessor


class BooleanModel:
    """
    Implements Boolean Information Retrieval Model.
    Supports AND, OR, NOT operators with postings list merging.
    Queries like: 'information AND retrieval NOT database'
    """

    def __init__(self, indexer):
        self.indexer = indexer
        self.preprocessor = Preprocessor()

    # ------------------------------------------------------------------ #
    #  QUERY PARSING                                                       #
    # ------------------------------------------------------------------ #

    def parse_query(self, query_string):
        """
        Parses a Boolean query string into a list of (operator, term) tuples.
        Handles: AND, OR, NOT operators.
        Default operator between consecutive terms is AND.
        Example: 'information AND retrieval NOT database'
        → [('AND', 'information'), ('AND', 'retrieval'), ('NOT', 'database')]
        """
        tokens = query_string.strip().upper().split()
        parsed = []
        current_op = 'AND'
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in ('AND', 'OR', 'NOT'):
                current_op = token
            else:
                term = token.lower()
                parsed.append((current_op, term))
                current_op = 'AND'  # reset to default
            i += 1
        return parsed

    # ------------------------------------------------------------------ #
    #  POSTINGS LIST OPERATIONS                                            #
    # ------------------------------------------------------------------ #

    def _get_doc_ids(self, term):
        """
        Gets sorted list of doc_ids containing the given term.
        Preprocesses the term before lookup.
        """
        proc_tokens = self.preprocessor.preprocess(term)
        if not proc_tokens:
            return set()
        proc_term = proc_tokens[0]
        postings = self.indexer.inverted_index.get(proc_term, [])
        return set(p['doc_id'] for p in postings)

    def _get_all_doc_ids(self):
        """Returns the set of all document IDs in the collection."""
        return set(doc['id'] for doc in self.indexer.documents)

    def _and_merge(self, set_a, set_b):
        """
        AND merge: returns documents in BOTH sets.
        Equivalent to set intersection.
        """
        return set_a & set_b

    def _or_merge(self, set_a, set_b):
        """
        OR merge: returns documents in EITHER set.
        Equivalent to set union.
        """
        return set_a | set_b

    def _not_merge(self, all_docs, set_b):
        """
        NOT: returns all documents NOT in set_b.
        Equivalent to set difference from the full collection.
        """
        return all_docs - set_b

    # ------------------------------------------------------------------ #
    #  BOOLEAN RETRIEVAL                                                   #
    # ------------------------------------------------------------------ #

    def retrieve(self, query_string):
        """
        Processes a Boolean query and returns matching document IDs.
        Supports AND, OR, NOT operators.
        """
        if not query_string.strip():
            return []

        parsed = self.parse_query(query_string)
        all_doc_ids = self._get_all_doc_ids()

        if not parsed:
            return []

        # Initialize result set with documents of first term
        first_op, first_term = parsed[0]
        result_set = self._get_doc_ids(first_term)
        if first_op == 'NOT':
            result_set = self._not_merge(all_doc_ids, result_set)

        # Apply remaining operators sequentially
        for op, term in parsed[1:]:
            term_docs = self._get_doc_ids(term)
            if op == 'AND':
                result_set = self._and_merge(result_set, term_docs)
            elif op == 'OR':
                result_set = self._or_merge(result_set, term_docs)
            elif op == 'NOT':
                result_set = self._not_merge(result_set, term_docs)

        return sorted(list(result_set))

    def retrieve_with_docs(self, query_string):
        """
        Performs Boolean retrieval and returns full document objects.
        """
        doc_ids = self.retrieve(query_string)
        results = []
        for doc_id in doc_ids:
            doc = self.indexer.get_doc_by_id(doc_id)
            if doc:
                results.append({
                    'id': doc['id'],
                    'topic': doc['topic'],
                    'unit': doc['unit'],
                    'content': doc['content'][:250] + '...' if len(doc['content']) > 250 else doc['content'],
                    'score': 1.0,
                    'model': 'Boolean'
                })
        return results

    def explain_query(self, query_string):
        """
        Explains how the Boolean query is interpreted step by step.
        Returns a list of explanation strings for display in the UI.
        """
        parsed = self.parse_query(query_string)
        explanations = []
        explanations.append(f"Parsed {len(parsed)} terms from query: '{query_string}'")
        for op, term in parsed:
            proc = self.preprocessor.preprocess(term)
            proc_str = proc[0] if proc else term
            docs = self._get_doc_ids(term)
            explanations.append(
                f"  {op} '{term}' (processed: '{proc_str}') → {len(docs)} document(s)"
            )
        final = self.retrieve(query_string)
        explanations.append(f"Final Boolean result: {len(final)} matching document(s)")
        return explanations