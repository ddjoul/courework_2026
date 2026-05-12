# main.py
from src.preprocessing import load_data, clean_data, sort_data, encode_ids, create_reward
from src.environment.recommender_env import RecommenderEnv

from src.models.bandit.epsilon_greedy import EpsilonGreedy
#from src.models.bandit.thompson_sampling import ThompsonSampling
from src.models.bandit.ucb import UCB
from src.models.bandit.linear_thompson import SimpleLinearTS
from src.models.random_model import RandomModel
from src.models.popularity_model import PopularityRecommender

from src.evaluation.evaluator import Evaluator

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt


def print_progress(current, total, model_name="", start_time=None):
    """Выводит прогресс-бар и оставшееся время."""
    percent = (current / total) * 100
    bar_length = 30
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    status = f"|{bar}| {percent:.1f}% ({current}/{total})"
    
    if start_time and current > 0:
        elapsed = time.time() - start_time
        eta = (elapsed / current) * (total - current)
        if eta < 60:
            status += f" | ETA: {eta:.0f}s"
        else:
            status += f" | ETA: {eta/60:.1f}min"
    
    if model_name:
        status = f"[{model_name}] {status}"
    
    print(f"\r{status}", end="", flush=True)


def run():
    start_total = time.time()
    
    os.makedirs("result/logs", exist_ok=True)
    os.makedirs("result/plots", exist_ok=True)
    
    
    # Data
    print("Loading data...")
    df = load_data()
    df = clean_data(df)
    df = sort_data(df)
    df, _, _ = encode_ids(df)
    df = create_reward(df)
    
    print(f"Total unique items: {df['item_id_enc'].nunique()}")
    print(f"Total interactions: {len(df)}")
    print()
    
    
    # Env
    env = RecommenderEnv(df)
    n_items = df["item_id_enc"].nunique()
    
    
    # Models
    models = {
        "random": RandomModel(n_items),
        "popularity": PopularityRecommender(),
        "epsilon_greedy": EpsilonGreedy(list(range(n_items)), epsilon=0.05),
        "ucb": UCB(list(range(n_items)), c=0.3),
        "simple_linear_ts": SimpleLinearTS(n_items=n_items, n_features=10)
    }
    
    results = {}
    total_models = len(models)
    
    
    # Train & Evaluate
    print(f"Starting evaluation of {total_models} models...")
    print("=" * 60)
    
    for idx, (name, model) in enumerate(models.items(), 1):
        model_start = time.time()
        
        print(f"\n[{idx}/{total_models}] Running {name}...")
        print("-" * 40)
        
        # Fit model
        print("  Fitting model...")
        fit_start = time.time()
        model.fit(df)
        print(f"  Fit completed in {time.time() - fit_start:.1f}s")
        
        # Evaluate
        print("  Evaluating...")
        eval_start = time.time()
        evaluator = Evaluator(env, model)
        results[name] = evaluator.evaluate(n_episodes=5)
        eval_time = time.time() - eval_start
        print(f"  Evaluation completed in {eval_time:.1f}s")
        
        # Save results
        with open(f"result/logs/{name}.json", "w") as f:
            json.dump(results[name], f, indent=4)
        
        model_time = time.time() - model_start
        print(f"  {name} - CTR: {results[name]['ctr']:.4f}, Avg Reward: {results[name]['avg_reward']:.4f}")
        print(f"  Time: {model_time:.1f}s")
        
        # Общий прогресс
        elapsed_total = time.time() - start_total
        print(f"\n  Overall progress: {idx}/{total_models} models done ({elapsed_total:.0f}s elapsed)")
    
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    for k, v in results.items():
        print(f"{k:20s} | CTR: {v['ctr']:.4f} | Avg Reward: {v['avg_reward']:.4f} | Impressions: {v['impressions']}")
    
    total_time = time.time() - start_total
    print(f"\nTotal execution time: {total_time:.0f}s ({total_time/60:.1f}min)")
    
    
    # Plot results
    print("\nGenerating plots...")
    plot_start = time.time()
    
    # Plot reward over time for each model
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        if "reward_history" in res:
            rewards = res["reward_history"]
            if len(rewards) > 100:
                window = min(50, len(rewards) // 10)
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(smoothed, label=name, alpha=0.8)
            else:
                plt.plot(rewards, label=name, alpha=0.8)
    
    plt.title("Reward over time (smoothed)", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("result/plots/reward_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Individual plots
    for name, res in results.items():
        plt.figure(figsize=(10, 5))
        plt.plot(res["reward_history"], linewidth=1)
        plt.title(f"Reward over time - {name}", fontsize=14)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"result/plots/reward_{name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ reward_{name}.png")
    
    # CTR comparison bar chart
    names = list(results.keys())
    ctr_values = [results[n]["ctr"] for n in names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, ctr_values, color=['gray', 'blue', 'orange', 'red', 'green'])
    plt.title("CTR Comparison - All Models", fontsize=14)
    plt.ylabel("CTR (Click-Through Rate)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for bar, ctr in zip(bars, ctr_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ctr:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("result/plots/ctr_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots generated in {time.time() - plot_start:.1f}s")
    print("\nDone!")


if __name__ == "__main__":
    run()