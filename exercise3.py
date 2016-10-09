from __future__ import print_function
import gym
import random

def q_frozen_lake(gamma=0.99, alpha=0.1, epsilon=0.1):
    """

    :param gamma: discount factor
    :param alpha: learning rate
    :return:
    """
    env = gym.make('FrozenLake-v0')

    total_reward = 0.0
    total_times_lost = 0
    total_time_won = 0

    number_of_episodes = 100
    iterations_per_episode = 100

    # Q(s, a) represented by dictionary of lists.
    # Key=state, value=list of 4 values for each direction
    # 0=left, 1=down, 2=right, 3=up
    q = {}

    for episode_count in range(number_of_episodes):
        state = env.reset()  # Starts episode in state 0
        for t in range(iterations_per_episode):
            print("Episode:", episode_count, "Time step:", t)
            env.render()
            print("")
            pre_move_state = state
            if state not in q:
                q[state] = [0 for _ in range(4)]
            action = pick_random_action_among_max(q[state])
            state, reward, done, info = env.step(action)

            # Update step
            if state not in q:
                q[state] = q[state] = [0 for _ in range(4)]

            q[pre_move_state][action] += alpha * (reward + gamma * max(q[state]) - q[pre_move_state][action])

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                if reward == 1.0:
                    print("FOUND THE GOAL. Reward:", reward)
                    total_time_won += 1
                    total_reward += reward
                    env.render()
                else:
                    print("Fell into the hole. Reward: ", reward)
                    total_times_lost += 1
                    env.render()
                break

    print("\nNumber of episodes executed:", number_of_episodes)
    print("Total times won:", total_time_won)
    print("Total times lost:", total_times_lost)
    print("Average reward:", total_reward / number_of_episodes)
    print(q)

def pick_random_action_among_max(actions):
    """

    :param actions: list with q values for each action where index corresponds to action
    :return:
    """
    max_q_value = max(actions)
    max_actions = []
    for a in range(len(actions)):
        if actions[a] == max_q_value:
            max_actions.append(a)

    return random.choice(max_actions)

q_frozen_lake()