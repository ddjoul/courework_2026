import os
from datasets import load_dataset
import pandas as pd


CACHE_PATH = "data/raw/cache.csv"


def load_yambda(sample_size=50000):
    # Load from cache
    if os.path.exists(CACHE_PATH):
        print("Loading from cache...")
        df = pd.read_csv(CACHE_PATH)

        print("Cache loaded:")
        print(df.shape)

        return df

    # If cache not found, downloand from site
    ds = load_dataset(
        "yandex/yambda",
        "flat-multievent-5b",
        streaming=True
    )

    data = []

    for i, row in enumerate(ds["train"]):
        data.append(row)

        if i + 1 >= sample_size:
            break

    df = pd.DataFrame(data)

    # Drop of empty and redundant data
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    # Keeping only core user interaction events
    if "event_type" in df.columns:
        valid_events = ["listen", "skip", "like", "dislike"]
        df = df[df["event_type"].isin(valid_events)]

    # Save
    os.makedirs("data/raw", exist_ok=True)

    df.to_csv(CACHE_PATH, index=False)
    print(f"Saved cache to {CACHE_PATH}")

    # Debug
    print("\nFinal dataset:")
    print(df.shape)
    print(df.head())

    return df


if __name__ == "__main__":
    df = load_yambda()