import numpy as np
from .base_recommender import BaseRecommender


class RandomModel(BaseRecommender):
    def __init__(self, n_items):
        self.n_items = n_items

    def fit(self, df):
        return self

    def recommend(self, user_id, state=None, k=10):
        return np.random.randint(0, self.n_items, size=k)