import numpy as np
from collections import defaultdict
from ..base_recommender import BaseRecommender
import random

class EpsilonGreedy:
    def __init__(self, items, epsilon=0.1):
        self.items = list(items)
        self.epsilon = epsilon

        self.q_values = {item: 0.0 for item in self.items}
        self.counts = {item: 0 for item in self.items}

    def fit(self, df):
        return self

    def recommend(self, user_id=None, state=None, k=1):
        if random.random() < self.epsilon:
            return random.choice(self.items)

        return max(self.q_values, key=self.q_values.get)

    def update(self, item, reward):
        self.counts[item] += 1
        n = self.counts[item]

        self.q_values[item] += (reward - self.q_values[item]) / n