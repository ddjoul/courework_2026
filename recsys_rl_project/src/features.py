import numpy as np
import pandas as pd


class FeatureBuilder:
    """
    Преобразует сырые логи в признаки для contextual bandit / RL моделей.
    """

    def __init__(self, history_size: int = 5):
        self.history_size = history_size

    def build_user_features(self, df: pd.DataFrame):
        """
        Простые user features:
        - средний reward
        - активность
        """
        user_stats = df.groupby("user_id").agg(
            user_activity=("item_id_enc", "count"),
            avg_reward=("reward", "mean")
        ).reset_index()

        return user_stats

    def build_item_features(self, df: pd.DataFrame):
        """
        Item popularity features
        """
        item_stats = df.groupby("item_id_enc").agg(
            item_popularity=("user_id", "count"),
            avg_reward=("reward", "mean")
        ).reset_index()

        return item_stats

    def build_state(self, user_history: list):
        """
        State = последние N айтемов пользователя
        """
        if len(user_history) < self.history_size:
            pad = [0] * (self.history_size - len(user_history))
            return np.array(pad + user_history)

        return np.array(user_history[-self.history_size:])

    def build_training_matrix(self, df: pd.DataFrame):
        """
        Формирует (state, action, reward)-подобную структуру
        """
        data = []

        for user_id, group in df.groupby("user_id"):
            group = group.sort_values("timestamp")

            history = []

            for _, row in group.iterrows():
                state = self.build_state(history)
                action = row["item_id_enc"]
                reward = row["reward"]

                data.append({
                    "user_id": user_id,
                    "state": state.tolist(),
                    "action": action,
                    "reward": reward
                })

                history.append(action)

        return pd.DataFrame(data)