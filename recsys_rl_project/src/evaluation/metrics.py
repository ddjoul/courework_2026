import numpy as np


class Metrics:
    @staticmethod
    def average_reward(rewards):
        return float(np.mean(rewards)) if len(rewards) > 0 else 0.0

    @staticmethod
    def ctr(clicks, impressions):
        if impressions == 0:
            return 0.0
        return clicks / impressions

    @staticmethod
    def hit_rate(recommended, actual):
        """
        recommended: list
        actual: single item or list
        """
        if isinstance(actual, list):
            return int(any(a in recommended for a in actual))
        return int(actual in recommended)

    @staticmethod
    def coverage(recommended_items, n_total_items):
        return len(set(recommended_items)) / n_total_items

    @staticmethod
    def ndcg(recommended, actual, k=10):
        """
        Simplified NDCG@k
        """
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item == actual:
                dcg = 1.0 / np.log2(i + 2)
                break

        idcg = 1.0
        return dcg / idcg if idcg > 0 else 0.0