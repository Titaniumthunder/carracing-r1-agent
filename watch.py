import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import pygame
import sys

def make_env():
    env = DummyVecEnv([lambda: gym.make("CarRacing-v3", render_mode="human")])
    env = VecFrameStack(env, n_stack=4)
    return env

def main():
    model = PPO.load("/Users/alexsalamati/racing-bot/carracing_ppo")
    env = make_env()
    obs = env.reset()

    pygame.init()
    font = pygame.font.SysFont("Arial", 24)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs = env.reset()
                    print("Restarted episode")

        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

    env.close()
    pygame.quit()
    sys.exit()

main()