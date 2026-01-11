import gymnasium
import trp_env

env = gymnasium.make("trp_env:SmallAntTRP-v1", n_blue=14, n_red=8, on_texture=False, vision=True, width=640, height=480)

obs, info = env.reset(seed=42)

done = False

try:
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
except KeyboardInterrupt:
    env.close()