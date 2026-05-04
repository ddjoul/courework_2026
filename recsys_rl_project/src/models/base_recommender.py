import numpy as np


class BaseRecommender:
    """
    Базовый интерфейс для всех рекомендательных моделей.
    """

    def fit(self, df):
        raise NotImplementedError

    def recommend(self, user_id, state=None, k=10):
        raise NotImplementedError
    
    def update(self, action, reward):
        pass  # default: no learning


class RandomRecommender(BaseRecommender):
    def __init__(self, n_items):
        self.n_items = n_items

    def fit(self, df):
        return self

    def recommend(self, user_id, state=None, k=10):
        return np.random.choice(self.n_items, size=k, replace=False)