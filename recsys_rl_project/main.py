# main.py
from src.preprocessing import load_data, clean_data, sort_data, encode_ids, create_reward
from src.environment.recommender_env import RecommenderEnv

from src.models.bandit.epsilon_greedy import EpsilonGreedy
#from src.models.bandit.thompson_sampling import ThompsonSampling
from src.models.bandit.ucb import UCB
from src.models.bandit.linear_thompson import SimpleLinearTS  # ADD THIS
from src.models.random_model import RandomModel
from src.models.popularity_model import PopularityRecommender

from src.evaluation.evaluator import Evaluator

import os
import json
import matplotlib.pyplot as plt


def run():
    os.makedirs("result/logs", exist_ok=True)
    os.makedirs("result/plots", exist_ok=True)
    
    # -----------------------------
    # Data
    # -----------------------------
    df = load_data()
    df = clean_data(df)
    df = sort_data(df)
    df, _, _ = encode_ids(df)
    df = create_reward(df)
    
    # OPTIONAL: Filter to reduce action space for better performance
    print(f"Total unique items: {df['item_id_enc'].nunique()}")
    
    # Filter to top items if needed (uncomment if too slow)
    # top_items = df['item_id_enc'].value_counts().head(10000).index
    # df = df[df['item_id_enc'].isin(top_items)]
    # print(f"Filtered to top {len(top_items)} items")
    
    # -----------------------------
    # Env
    # -----------------------------
    env = RecommenderEnv(df)
    n_items = df["item_id_enc"].nunique()
    
    # -----------------------------
    # Models
    # -----------------------------
    models = {
        "random": RandomModel(n_items),
        "popularity": PopularityRecommender(),
        "epsilon_greedy": EpsilonGreedy(list(range(n_items)), epsilon=0.05),
        "ucb": UCB(list(range(n_items)), c=0.3),  # было 2.0
        # "thompson": ThompsonSampling(list(range(min(n_items, 5000)))),  # Original TS (slow)
        "simple_linear_ts": SimpleLinearTS(n_items=n_items, n_features=10)
    }
    
    results = {}
    
    # -----------------------------
    # Train & Evaluate
    # -----------------------------
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Running {name}...")
        print(f"{'='*50}")
        
        # Fit model
        model.fit(df)
        
        # Evaluate
        evaluator = Evaluator(env, model)
        results[name] = evaluator.evaluate(n_episodes=5)  # Увеличьте для лучших результатов
        
        # Save results
        with open(f"result/logs/{name}.json", "w") as f:
            json.dump(results[name], f, indent=4)
        
        print(f"{name} - CTR: {results[name]['ctr']:.4f}, Avg Reward: {results[name]['avg_reward']:.4f}")
    
    # -----------------------------
    # Print results
    # -----------------------------
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    for k, v in results.items():
        print(f"{k:20s} | CTR: {v['ctr']:.4f} | Avg Reward: {v['avg_reward']:.4f} | Impressions: {v['impressions']}")
    
    # -----------------------------
    # Plot results
    # -----------------------------
    # Plot reward over time for each model
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        if "reward_history" in res:
            # Smooth rewards for better visualization
            rewards = res["reward_history"]
            if len(rewards) > 100:
                # Moving average
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
    
    # CTR comparison bar chart
    names = list(results.keys())
    ctr_values = [results[n]["ctr"] for n in names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, ctr_values, color=['gray', 'blue', 'orange', 'red', 'green'])
    plt.title("CTR Comparison - All Models", fontsize=14)
    plt.ylabel("CTR (Click-Through Rate)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, ctr in zip(bars, ctr_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ctr:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("result/plots/ctr_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    


if __name__ == "__main__":
    run()