"""Test Gymnasium/Shimmy integration with DeepMind Lab."""

import deepmind_lab
from shimmy.dm_lab_compatibility import DmLabCompatibilityV0

env = deepmind_lab.Lab(
    "lt_chasm",
    ["RGBD"],
    config={"width": "640", "height": "480"},
    renderer="software",
)
gym_env = DmLabCompatibilityV0(env)

obs, info = gym_env.reset(seed=42)
print(f"Observation space: {gym_env.observation_space}")
print(f"Action space: {gym_env.action_space}")

for _ in range(10):
    action = gym_env.action_space.sample()
    obs, reward, terminated, truncated, info = gym_env.step(action)
    if terminated or truncated:
        obs, info = gym_env.reset()

gym_env.close()
print("Gymnasium integration working.")
