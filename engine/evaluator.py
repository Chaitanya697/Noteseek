"""
evaluator.py
------------
Unit 2 — Retrieval Metrics and Performance Evaluation
Covers: Precision, Recall, F-Measure (F1, F-beta),
        Mean Average Precision (MAP), Precision@K, NDCG
"""
import math

class Evaluator:
    """
    Evaluates IR system effectiveness using standard metrics:
    Precision, Recall, F-Measure, MAP, and Precision@K.
    All metrics defined per Unit 2 of the IRT syllabus.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    #  UNIT 2 — PRECISION                                                 #
    # ------------------------------------------------------------------ #

    def precision(self, retrieved_ids, relevant_ids):
        """
        Precision = |Retrieved ∩ Relevant| / |Retrieved|
        Fraction of retrieved documents that are relevant.
        High precision → most results shown are useful.
        """
        if not retrieved_ids:
            return 0.0
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        tp = len(retrieved_set & relevant_set)
        return round(tp / len(retrieved_set), 4)

    # ------------------------------------------------------------------ #
    #  UNIT 2 — RECALL                                                    #
    # ------------------------------------------------------------------ #

    def recall(self, retrieved_ids, relevant_ids):
        """
        Recall = |Retrieved ∩ Relevant| / |Relevant|
        Fraction of relevant documents that were retrieved.
        High recall → most relevant results were found.
        """
        if not relevant_ids:
            return 0.0
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        tp = len(retrieved_set & relevant_set)
        return round(tp / len(relevant_set), 4)

    # ------------------------------------------------------------------ #
    #  UNIT 2 — F-MEASURE                                                 #
    # ------------------------------------------------------------------ #

    def f_measure(self, retrieved_ids, relevant_ids, beta=1.0):
        """
        F-Measure = (1 + β²) * P * R / (β² * P + R)
        Harmonic mean of Precision and Recall.
        β=1 (F1): equal weight to P and R
        β=2: recall weighted 2x more than precision
        β=0.5: precision weighted 2x more than recall
        """
        p = self.precision(retrieved_ids, relevant_ids)
        r = self.recall(retrieved_ids, relevant_ids)
        if p + r == 0:
            return 0.0
        beta_sq = beta ** 2
        f = (1 + beta_sq) * p * r / (beta_sq * p + r)
        return round(f, 4)

    def f1(self, retrieved_ids, relevant_ids):
        """Convenience method: F1 score (beta=1)."""
        return self.f_measure(retrieved_ids, relevant_ids, beta=1.0)

    # ------------------------------------------------------------------ #
    #  PRECISION AT K                                                      #
    # ------------------------------------------------------------------ #

    def precision_at_k(self, ranked_ids, relevant_ids, k):
        """
        Precision@K: precision considering only the top-K retrieved results.
        Important for measuring quality of top search results.
        """
        if not ranked_ids or k <= 0:
            return 0.0
        top_k = ranked_ids[:k]
        return self.precision(top_k, relevant_ids)

    # ------------------------------------------------------------------ #
    #  AVERAGE PRECISION & MAP                                             #
    # ------------------------------------------------------------------ #

    def average_precision(self, ranked_ids, relevant_ids):
        """
        Average Precision (AP): averages precision at each recall point
        where a relevant document is found.
        AP = (1/|Relevant|) * Σ P@k * rel(k)
        """
        if not relevant_ids or not ranked_ids:
            return 0.0
        relevant_set = set(relevant_ids)
        hits = 0
        sum_precision = 0.0
        for k, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in relevant_set:
                hits += 1
                sum_precision += hits / k
        if hits == 0:
            return 0.0
        return round(sum_precision / len(relevant_set), 4)

    def mean_average_precision(self, results_per_query):
        """
        Mean Average Precision (MAP): mean of AP across multiple queries.
        results_per_query: list of (ranked_ids, relevant_ids) tuples.
        """
        if not results_per_query:
            return 0.0
        ap_scores = [
            self.average_precision(ranked, relevant)
            for ranked, relevant in results_per_query
        ]
        return round(sum(ap_scores) / len(ap_scores), 4)

    # ------------------------------------------------------------------ #
    #  NDCG — NORMALIZED DISCOUNTED CUMULATIVE GAIN                       #
    # ------------------------------------------------------------------ #

    def dcg(self, ranked_ids, relevant_ids, k=None):
        """
        Discounted Cumulative Gain (DCG):
        DCG = Σ rel(i) / log2(i+1)
        Documents ranked higher contribute more to the score.
        """
        relevant_set = set(relevant_ids)
        ranked = ranked_ids[:k] if k else ranked_ids
        dcg_score = 0.0
        for i, doc_id in enumerate(ranked, start=1):
            rel = 1 if doc_id in relevant_set else 0
            dcg_score += rel / math.log2(i + 1)
        return dcg_score

    def ndcg(self, ranked_ids, relevant_ids, k=None):
        """
        Normalized DCG = DCG / IDCG
        IDCG is the ideal DCG (all relevant docs ranked first).
        Score between 0 and 1.
        """
        ideal_ranked = list(set(relevant_ids)) + \
                       [i for i in ranked_ids if i not in set(relevant_ids)]
        idcg = self.dcg(ideal_ranked, relevant_ids, k)
        if idcg == 0:
            return 0.0
        return round(self.dcg(ranked_ids, relevant_ids, k) / idcg, 4)

    # ------------------------------------------------------------------ #
    #  FULL EVALUATION REPORT                                             #
    # ------------------------------------------------------------------ #

    def evaluate(self, retrieved_ids, relevant_ids, ranked_ids=None):
        """
        Full evaluation report for a single query result set.
        Returns all metrics in a structured dictionary.
        """
        if ranked_ids is None:
            ranked_ids = retrieved_ids

        p = self.precision(retrieved_ids, relevant_ids)
        r = self.recall(retrieved_ids, relevant_ids)
        f1 = self.f1(retrieved_ids, relevant_ids)
        f2 = self.f_measure(retrieved_ids, relevant_ids, beta=2.0)
        f05 = self.f_measure(retrieved_ids, relevant_ids, beta=0.5)
        ap = self.average_precision(ranked_ids, relevant_ids)
        p5 = self.precision_at_k(ranked_ids, relevant_ids, k=5)
        p10 = self.precision_at_k(ranked_ids, relevant_ids, k=10)
        ndcg_score = self.ndcg(ranked_ids, relevant_ids)

        # Confusion matrix components
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        tp = len(retrieved_set & relevant_set)
        fp = len(retrieved_set - relevant_set)
        fn = len(relevant_set - retrieved_set)

        return {
            'precision': p,
            'recall': r,
            'f1_score': f1,
            'f2_score': f2,
            'f0_5_score': f05,
            'average_precision': ap,
            'precision_at_5': p5,
            'precision_at_10': p10,
            'ndcg': ndcg_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'retrieved_count': len(retrieved_ids),
            'relevant_count': len(relevant_ids)
        }

    def precision_recall_curve(self, ranked_ids, relevant_ids):
        """
        Computes the precision-recall curve as recall increases.
        Returns list of (recall, precision) points for plotting.
        """
        relevant_set = set(relevant_ids)
        hits = 0
        curve = []
        for k, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in relevant_set:
                hits += 1
            p = hits / k
            r = hits / len(relevant_set) if relevant_set else 0
            curve.append({'rank': k, 'precision': round(p, 4), 'recall': round(r, 4)})
        return curve

    def interpolated_precision(self, pr_curve, recall_levels=None):
        """
        11-point interpolated precision-recall curve.
        Recall levels: [0.0, 0.1, 0.2, ..., 1.0]
        """
        if recall_levels is None:
            recall_levels = [i / 10 for i in range(11)]
        interpolated = []
        for level in recall_levels:
            # Max precision at or above this recall level
            p_at_level = max(
                (pt['precision'] for pt in pr_curve if pt['recall'] >= level),
                default=0.0
            )
            interpolated.append({'recall': level, 'precision': round(p_at_level, 4)})
        return interpolated