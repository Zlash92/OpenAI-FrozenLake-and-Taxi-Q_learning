from __future__ import print_function
import gym

env = gym.make('Taxi-v2')
print("Observation space", env.observation_space)
print("Action space", env.action_space)

total_reward = 0.0
total_times_lost = 0
total_time_won = 0

number_of_episodes = 10
numer_of_iterations = 100
for episode_count in range(number_of_episodes):
    state = env.reset()
    for t in range(1000):
        print("Episode:", episode_count, "Time step:", t)
        env.render()
        print("")
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            if reward == 20:
                total_time_won += 1
            break

print("\nNumber of episodes executed:", number_of_episodes)
print("Total times passenger dropped off:", total_time_won)

