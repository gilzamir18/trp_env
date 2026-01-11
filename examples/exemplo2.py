import gymnasium
import trp_env
import gc

env = gymnasium.make("SmallLowGearAntTRP-v1", mode="rgb_array", camera_id=0, vision=True, width=64, height=64)

env.reset(seed=42)

# RGB (0-255, size=64x64x3)
vision = env.render()

# RGBD (0-255, size=64x64x4, depth values are also normalized into 0-255)
vision = env.render()
