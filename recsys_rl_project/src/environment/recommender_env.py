import numpy as np
import pandas as pd


class RecommenderEnv:
    """
    Простая RL/Contextual Bandit среда для рекомендательной системы.

    Идея:
    - состояние = последние N айтемов пользователя
    - действие = рекомендованный item
    - награда = reward (played_ratio или shaped reward)
    """

    def __init__(self, df: pd.DataFrame, history_size: int = 5):
        """
        df должен содержать:
        - user_id
        - item_id_enc
        - reward
        - timestamp (желательно)
        """
        self.df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        self.history_size = history_size

        self.users = self.df["user_id"].unique()

        # сгруппируем данные по пользователям
        self.user_data = {
            u: group.reset_index(drop=True)
            for u, group in self.df.groupby("user_id")
        }

        self.reset()

    # -----------------------------
    # reset environment
    # -----------------------------
    def reset(self, user_id=None):
        """
        Начинаем эпизод (по пользователю)
        """
        if user_id is None:
            self.current_user = np.random.choice(self.users)
        else:
            self.current_user = user_id

        self.user_df = self.user_data[self.current_user]
        self.t = self.history_size

        return self._get_state()

    # -----------------------------
    # state = last N items
    # -----------------------------
    def _get_state(self):
        history = self.user_df["item_id_enc"].values

        if self.t < self.history_size:
            pad = [0] * (self.history_size - self.t)
            state = pad + list(history[:self.t])
        else:
            state = history[self.t - self.history_size:self.t]

        return np.array(state, dtype=np.int32)

    # -----------------------------
    # step function
    # -----------------------------
    def step(self, action: int):
        """
        action = recommended item
        """

        # текущий реальный item пользователя
        if isinstance(action, (list, np.ndarray)):
            action = int(np.array(action).flatten()[0])

        action = int(action)

        true_item = int(self.user_df.iloc[self.t]["item_id_enc"])
        reward = float(self.user_df.iloc[self.t]["reward"])

        if action == true_item:
            reward *= 1.2

        self.t += 1

        done = self.t >= len(self.user_df)
        next_state = self._get_state() if not done else None

        return next_state, reward, done, {}

    # -----------------------------
    # number of items (for agent)
    # -----------------------------
    def get_action_space_size(self):
        return self.df["item_id_enc"].nunique()

    # -----------------------------
    # sample random action
    # -----------------------------
    def sample_action(self):
        return np.random.randint(0, self.get_action_space_size())


# -----------------------------
# quick sanity check
# -----------------------------
if __name__ == "__main__":
    from preprocessing import load_data, clean_data, sort_data, encode_ids, create_reward

    df = load_data()
    df = clean_data(df)
    df = sort_data(df)
    df, _, _ = encode_ids(df)
    df = create_reward(df)

    env = RecommenderEnv(df)

    state = env.reset()
    print("Initial state:", state)

    for _ in range(10):
        action = env.sample_action()
        next_state, reward, done, _ = env.step(action)

        print("action:", action, "reward:", reward)

        if done:
            break