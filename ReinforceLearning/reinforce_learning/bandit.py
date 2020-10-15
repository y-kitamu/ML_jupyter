"""N本腕banditの実装
"""
import numpy as np
from tqdm import tqdm


class NArmBandit:
    """N本腕バンディット。
    各行動(a)の報酬Q(a)は、平均Q*(a) (真の報酬), 分散1のガウス分布に従っているとする
    Args:
        n (int) : 腕(行動)の数
        std_dev0 (float) : 各行動の真の報酬の分布の標準偏差
        std_dev1 (float) : 各試行における報酬の分布の標準偏差
    """
    def __init__(self, n=10, std_dev0=1.0, std_dev1=1.0, **kwargs):
        self.arms = 10
        self.true_rewards = np.random.normal(0.0, std_dev0, self.arms)
        self.std_dev0 = std_dev0
        self.std_dev1 = std_dev1

    def get_reward(self, index):
        self.update_rewards()
        return np.random.normal(self.true_rewards[index], self.std_dev1)

    def update_rewards(self):
        return


class RandomWalkNArmBandit(NArmBandit):
    """rewardがRandomWalkで変動する
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_rewards()

    def initialize_rewards(self):
        self.true_rewards = np.ones(self.true_rewards.shape)

    def update_rewards(self):
        self.true_rewards += np.random.normal(0.0, self.std_dev0, self.arms)


class Agent:
    """M個のN本腕バンディット環境。
    Args:
        m (int)
        n (int)
    """
    def __init__(self, m=2000, n=10, bandit=NArmBandit, **kwargs):
        self.bandits = [bandit(n, **kwargs) for _ in range(m)]
        self.rewards = np.zeros(m)
        self.sample_rewards = [[[] for _ in range(n)] for _ in range(m)]
        self.sample_counts = np.zeros((m, n))
        self.sample_mean_reward = np.zeros((m, n))

    def reset(self):
        self.sample_rewards = [[[] for _ in range(len(self.sample_rewards[0]))]
                               for _ in range(len(self.sample_rewards))]
        self.sample_mean_reward[:] = 0

    def get_rewards_naive(self, selected_indices):
        total_reward = 0
        for i, selected_idx in enumerate(selected_indices):
            reward = self.bandits[i].get_reward(selected_idx)
            self.sample_rewards[i][selected_idx].append(reward)
            self.sample_mean_reward[i, selected_idx] = np.mean(
                self.sample_rewards[i][selected_idx])
            total_reward += reward
        return total_reward

    def get_rewards(self, selected_indices, alpha=None):
        total_reward = 0

        for i, selected_idx in enumerate(selected_indices):
            reward = self.bandits[i].get_reward(selected_idx)
            self.sample_counts[i][selected_idx] += 1
            _alpha = alpha if alpha is not None else 1.0 / self.sample_counts[i][selected_idx]
            self.sample_mean_reward[i, selected_idx] += _alpha * (
                reward - self.sample_mean_reward[i, selected_idx])
            total_reward += reward
            self.rewards[i] = reward
        return total_reward


class Client:
    """ある戦略に沿ってM個のN本腕バンディットの報酬を獲得する
    """
    def __init__(self, m=2000, n=10, k=1000, alpha=None, **kwargs):
        self.agent = Agent(m, n, **kwargs)
        self.rewards = []
        self.n = n
        self.m = m
        self.k = k
        self.alpha = alpha

    def run(self):
        """k回の試行を行う
        """
        self.agent.reset()
        self.rewards = []
        for i in tqdm(range(self.k)):
            reward = self.agent.get_rewards(self.select_indices(), self.alpha)
            self.rewards.append(reward / self.m)

    def select_indices(self):
        """m個のN本腕バンディットについて、それぞれどの腕を選択するかを計算する
        Return : list of int (バンディットで選択した腕のindexを記載した長さmの配列)
        """
        raise NotImplemented


class GreedyClient(Client):
    def select_indices(self):
        return self.agent.sample_mean_reward.argmax(axis=1)


class EpsilonGreedyClient(GreedyClient):
    def __init__(self, m=2000, n=10, k=1000, epsilon=0.1, **kwargs):
        super().__init__(m, n, k, **kwargs)
        self.epsilon = epsilon

    def select_indices(self):
        indices = super().select_indices()
        for idx, bandit in enumerate(self.agent.bandits):
            if np.random.rand() < self.epsilon:
                indices[idx] = np.random.randint(bandit.arms)
        return indices


class SoftmaxClient(Client):
    def __init__(self, m=2000, n=10, k=1000, tau=1.0, **kwargs):
        super().__init__(m, n, k, **kwargs)
        self.n = n
        self.tau = tau

    def select_indices(self):
        indices = np.zeros(self.m, dtype=int)
        softmax = np.exp(self.agent.sample_mean_reward / self.tau)
        softmax_denomis = softmax.sum(axis=1, keepdims=True)
        probs = softmax / softmax_denomis
        choices = [i for i in range(self.n)]
        for idx in range(self.m):
            indices[idx] = np.random.choice(choices, p=probs[idx])
        return indices


class ReinforcementComparisonClient(Client):
    def __init__(self, *args, alpha=0.1, beta=0.1, init_reward=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.priority = np.zeros((self.m, self.n))
        self.reference_rewards = np.ones(self.m) * init_reward

    def run(self):
        """k回の試行を行う
        """
        self.agent.reset()
        self.rewards = []
        rows_idx = np.array([i for i in range(self.m)])
        for i in tqdm(range(self.k)):
            indices = self.select_indices()
            reward = self.agent.get_rewards(indices, self.alpha)
            self.rewards.append(reward / self.m)
            where = (rows_idx, indices)
            self.priority[where] += self.beta * (self.agent.rewards - self.reference_rewards)
            self.reference_rewards += self.alpha * (self.agent.rewards - self.reference_rewards)

    def select_indices(self):
        softmax = np.exp(self.priority)
        probs = softmax / (softmax.sum(axis=1, keepdims=True) + 1e-5)
        probs /= probs.sum(axis=1, keepdims=True)

        choices = [i for i in range(self.n)]
        indices = np.zeros(self.m, dtype=int)
        for idx in range(self.m):
            indices[idx] = np.random.choice(choices, p=probs[idx])
        return indices


class PursitMethodClient(Client):
    def __init__(self, *args, beta=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.probs = np.ones((self.m, self.n)) * 1.0 / self.k

    def select_indices(self):
        greedy_idx = np.argmax(self.agent.sample_mean_reward, axis=1)
        where = (np.array([i for i in range(self.m)]), greedy_idx)
        self.probs += self.beta * (0 - self.probs)
        self.probs[where] += self.beta
        self.probs /= self.probs.sum(axis=1, keepdims=True)

        choices = [i for i in range(self.n)]
        indices = np.zeros(self.m, dtype=int)
        for idx in range(self.m):
            indices[idx] = np.random.choice(choices, p=self.probs[idx])
        return indices
