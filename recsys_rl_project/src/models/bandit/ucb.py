import numpy as np
from ..base_recommender import BaseRecommender
import math

class UCB:
    def __init__(self, items, c=2.0):
        self.items = list(items)
        self.c = c

        self.q_values = {item: 0.0 for item in self.items}
        self.counts = {item: 0 for item in self.items}
        self.total = 0

    def fit(self, df):
        return self

    def recommend(self, user_id=None, state=None, k=1):
        self.total += 1

        ucb_scores = {}

        for item in self.items:
            if self.counts[item] == 0:
                return item

            bonus = self.c * math.sqrt(
                math.log(self.total) / self.counts[item]
            )

            ucb_scores[item] = self.q_values[item] + bonus

        return max(ucb_scores, key=ucb_scores.get)

    def update(self, item, reward):
        self.counts[item] += 1
        n = self.counts[item]

        self.q_values[item] += (reward - self.q_values[item]) / n