import gymnasium as gym
import highway_env
# env = gym.make('safe-highway-fast-v0', render_mode='human')
env = gym.make('safe-intersection-v0', render_mode='human')

obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = 1.5 # Your agent code here
    obs, reward, done, truncated, info = env.step(action)