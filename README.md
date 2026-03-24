# CarRacing RL Agent

A reinforcement learning agent trained to drive in OpenAI Gymnasium's CarRacing-v3 environment using PPO (Proximal Policy Optimization).

## Results
- Average reward: ~896 / 1000
- Peak reward: 924
- Trained for 4M+ timesteps

## How it works
The agent uses a custom CNN to process raw pixel frames from the game and outputs steering, acceleration, and braking actions. It was trained using PPO with frame stacking (4 frames) so the model can infer speed and direction.

## Setup
pip install gymnasium[box2d] stable-baselines3 pygame torch

## Train from scratch
python3 train_ppo.py

Training saves automatically to carracing_ppo.zip. Each run loads the previous model and continues improving.

## Watch it drive
python3 watch.py

Press Q to quit, R to restart the episode.

## Architecture
- Policy: Custom 3-layer CNN (32→64→128 filters) + 2-layer MLP (256→256)
- Algorithm: PPO with frame stacking (n_stack=4)
- Parallel environments: 64
- Hardware: Apple M5 Pro (MPS acceleration)
