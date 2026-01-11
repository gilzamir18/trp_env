import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "cpu"

env_args = dict(n_blue=14, n_red=8, on_texture=False, vision=True, width=64, height=64, render_mode="rgb_array", camera_id=0)

# Parallel environments
vec_env = make_vec_env("trp_env:SmallAntTRP-v1", n_envs=4, env_kwargs=env_args)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo_model")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_model", device=device)

obs = vec_env.reset()
vec_env.render_mode="human"
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()