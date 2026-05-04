# src/models/popularity_model.py
import numpy as np
import pandas as pd
from collections import Counter
from .base_recommender import BaseRecommender


class PopularityRecommender(BaseRecommender):
    def __init__(self):
        self.popularity = None
        self.items = None

    def fit(self, df: pd.DataFrame):
        counts = Counter(df["item_id_enc"])
        self.popularity = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        self.items = [i for i, _ in self.popularity]
        return self

    def recommend(self, user_id, state=None, k=10):
        return np.random.choice(self.items[:50], size=k, replace=False)
