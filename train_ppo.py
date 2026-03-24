import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import os

class BiggerCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = {}
        self.fig, self.ax = plt.subplots()

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            if i not in self.current_rewards:
                self.current_rewards[i] = 0
            self.current_rewards[i] += self.locals["rewards"][i]
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)
                print(f"Episode {len(self.episode_rewards):4d} | "
                      f"Reward: {ep_reward:8.2f} | "
                      f"Timestep: {self.num_timesteps:7d}")
                self.current_rewards[i] = 0
                self._update_plot()
        return True

    def _update_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("PPO CarRacing-v3 — Episode Reward")
        eps = list(range(1, len(self.episode_rewards) + 1))
        self.ax.plot(eps, self.episode_rewards, alpha=0.4, color="steelblue", label="Reward")
        if len(self.episode_rewards) >= 20:
            rolling = [
                sum(self.episode_rewards[max(0, i - 19):i + 1]) / min(20, i + 1)
                for i in range(len(self.episode_rewards))
            ]
            self.ax.plot(eps, rolling, color="orange", linewidth=2, label="20-ep mean")
        self.ax.legend()
        self.fig.savefig("/Users/alexsalamati/racing-bot/reward_plot.png", dpi=100)

env = make_vec_env("CarRacing-v3", n_envs=64)
env = VecFrameStack(env, n_stack=4)

policy_kwargs = dict(
    features_extractor_class=BiggerCNN,
    features_extractor_kwargs=dict(features_dim=512),
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
)

model_path = "/Users/alexsalamati/racing-bot/carracing_ppo.zip"

if os.path.exists(model_path):
    print("Loading existing model...")
    model = PPO.load(
        model_path,
        env=env,
        device="mps",
    )
    model.learning_rate = 1e-4
    model.n_epochs = 10
else:
    print("Starting fresh model...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=8192,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device="mps",
    )

callback = EpisodeRewardCallback()
print("Training PPO on CarRacing-v3 for 3,000,000 timesteps...\n")
model.learn(total_timesteps=3_000_000, callback=callback)

print(f"\nTraining complete. Total episodes: {len(callback.episode_rewards)}")
if callback.episode_rewards:
    print(f"Mean reward (last 10 eps): {sum(callback.episode_rewards[-10:]) / min(10, len(callback.episode_rewards)):.2f}")

model.save("/Users/alexsalamati/racing-bot/carracing_ppo")
print("Model saved to ~/racing-bot/carracing_ppo.zip")
print("Reward plot saved to ~/racing-bot/reward_plot.png")
env.close()
plt.close()