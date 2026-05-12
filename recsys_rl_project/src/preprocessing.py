import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


CACHE_PATH = "data/raw/cache.csv"
OUTPUT_PATH = "data/processed/preprocessed.csv"


# 1. Load data
def load_data(path: str = CACHE_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    return df


# 2. Basic cleaning
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()

    # keep only listen events (for your dataset)
    if "event_type" in df.columns:
        df = df[df["event_type"] == "listen"]

    return df


# 3. Sort by time (important for RL)
def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.sort_values(["uid", "timestamp"]).reset_index(drop=True)
    return df


# 4. Encode users and items
def encode_ids(df: pd.DataFrame):
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    df["user_id"] = user_enc.fit_transform(df["uid"])
    df["item_id_enc"] = item_enc.fit_transform(df["item_id"])

    return df, user_enc, item_enc


# 5. Reward shaping (critical for RL)
def create_reward(df):
    df["reward"] = df["played_ratio_pct"] / 100.0
    return df


# 6. Train-test split (time-based per user)
def train_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    train_list = []
    test_list = []

    for uid, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("timestamp")

        split_idx = int(len(user_df) * (1 - test_ratio))

        train_list.append(user_df.iloc[:split_idx])
        test_list.append(user_df.iloc[split_idx:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df


# 7. Build RL dataset format (state, action, reward)
def build_rl_format(df: pd.DataFrame, history_size: int = 5):
    data = []

    for uid, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("timestamp")

        items = user_df["item_id_enc"].values
        rewards = user_df["reward"].values

        for i in range(history_size, len(user_df)):
            state = items[i - history_size:i]
            action = items[i]
            reward = rewards[i]

            data.append({
                "user_id": uid,
                "state": state.tolist(),
                "action": action,
                "reward": reward
            })

    return pd.DataFrame(data)


# 8. Pipeline
def run_pipeline():
    df = load_data()
    df = clean_data(df)
    df = sort_data(df)
    df, user_enc, item_enc = encode_ids(df)
    df = create_reward(df)

    train_df, test_df = train_test_split(df)

    rl_train = build_rl_format(train_df)
    rl_test = build_rl_format(test_df)

    os.makedirs("data/processed", exist_ok=True)

    rl_train.to_csv("data/processed/train_rl.csv", index=False)
    rl_test.to_csv("data/processed/test_rl.csv", index=False)

    print("Done preprocessing")
    print("Train shape:", rl_train.shape)
    print("Test shape:", rl_test.shape)


if __name__ == "__main__":
    run_pipeline()