import numpy as np
from src.evaluation.metrics import Metrics


class Evaluator:
    """
    Offline evaluator for recommender / bandit models
    """

    def __init__(self, env, model):
        self.env = env
        self.model = model

    def evaluate(self, n_episodes=10):
        rewards_history = []
        hits = 0
        impressions = 0

        for _ in range(n_episodes):
            state = self.env.reset()
            done = False

            while not done:
                # -----------------------------
                # 1. action
                # -----------------------------
                action = self.model.recommend(
                    user_id=self.env.current_user,
                    state=state
                )

                # защита от list/np.array
                if isinstance(action, (list, np.ndarray)):
                    action = int(np.array(action).flatten()[0])

                # -----------------------------
                # 2. env step
                # -----------------------------
                next_state, reward, done, _ = self.env.step(action)

                # -----------------------------
                # 3. learning step
                # -----------------------------
                if hasattr(self.model, "update"):
                    self.model.update(action, reward)

                # -----------------------------
                # 4. logging
                # -----------------------------
                rewards_history.append(reward)

                impressions += 1

                # более корректный CTR proxy
                hits += int(reward > 0.5)

                state = next_state

        return {
            "avg_reward": Metrics.average_reward(rewards_history),
            "ctr": Metrics.ctr(hits, impressions),

            "reward_history": rewards_history,
            "impressions": impressions,
            "hits": hits
        }