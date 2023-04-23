import gymnasium as gym_env
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle
from ray.rllib.algorithms.dqn import DQNConfig

class CustomSimpleCorridor(gym_env.Env):
    def __init__(self, config=None):
        config = config or {}
        self.end_pos = config.get("corridor_length", 10)
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, 999.0, shape=(1,), dtype=np.float32)

    def set_corridor_length(self, length):
        self.end_pos = length
        print("Corridor length updated to {}".format(length))

    def reset(self, *, seed=None, options=None):
        self.cur_pos = 0.0
        return [self.cur_pos], {}

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1.0
        elif action == 1:
            self.cur_pos += 1.0
        done = truncated = self.cur_pos >= self.end_pos
        reward = -0.1
        if done:
            reward += 1
        return [self.cur_pos], reward, done, truncated, {}

algo_config = (
    DQNConfig()
    .environment(
        env=CustomSimpleCorridor,
    )
    .rollouts(num_rollout_workers=3)
)
algo_instance = algo_config.build()
avg_rewards = []

for i in range(50):
    train_results = algo_instance.train()
    print(train_results)
    avg_rewards.append(train_results['episode_reward_mean'])
    print(f"Iter: {i}; avg. reward={train_results['episode_reward_mean']}")

plt.plot(avg_rewards)
plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title('Progress of Training')
plt.show()
