from __future__ import print_function
import gym

env = gym.make('Taxi-v1')
print("Observation space", env.observation_space)
print("Action space", env.action_space)

total_reward = 0.0
total_times_lost = 0
total_time_won = 0

number_of_episodes = 5
for episode_count in range(number_of_episodes):
    state = env.reset()
    for t in range(100):
        print("Episode:", episode_count, "Time step:", t)
        env.render()
        print("")
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print("\nNumber of episodes executed:", number_of_episodes)
print("Total times won:", total_time_won)
print("Total times lost:", total_times_lost)
print("Average reward:", total_reward/number_of_episodes)

