from __future__ import print_function
import gym

env = gym.make('FrozenLake-v0')
# print("Observation space", env.observation_space)
# print("Action space", env.action_space)

total_reward = 0.0
total_times_lost = 0
total_time_won = 0

number_of_episodes = 100
number_of_iterations = 100
for episode_count in range(number_of_episodes):
    state = env.reset()
    for t in range(number_of_iterations):
        print("Episode:", episode_count, "Time step:", t)
        env.render()
        print("")
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            if reward == 1.0:
                print("FOUND THE GOAL. Reward:", reward)
                total_time_won += 1
                total_reward += reward
            else:
                print("Fell into the hole. Reward: ", reward)
                total_times_lost += 1
            break

print("\nNumber of episodes executed:", number_of_episodes)
print("Total times won:", total_time_won)
print("Total times lost:", total_times_lost)
print("Average reward:", total_reward/number_of_episodes)

