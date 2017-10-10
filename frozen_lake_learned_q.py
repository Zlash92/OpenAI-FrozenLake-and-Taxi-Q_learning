from __future__ import print_function
import gym
import random
import matplotlib.pyplot as plt
import pickle


def q_frozen_lake(gamma=0.99, alpha=0.2, epsilon=0.01):
    """

    :param gamma: discount factor
    :param alpha: learning rate
    :return:
    """
    env = gym.make('FrozenLake-v0')
    number_of_states = 16
    learned_q_function = pickle.load(open("q_epsilon001.p", "rb"))

    all_episode_q_max = []
    total_times_lost = 0
    total_times_won = 0

    number_of_episodes = 5000
    iterations_per_episode = 100

    # Index-> direction 0->left, 1->down, 2->right, 3->up
    q = {}
    for i in range(number_of_states):
        q[i] = [0, 0, 0, 0]

    for episode_count in range(number_of_episodes):
        print("\nStarting episode:", episode_count)
        state = env.reset()  # Starts episode in state 0
        q_max_total = 0.0
        for t in range(iterations_per_episode):
            # print("Episode:", episode_count, "Time step:", t)
            # env.render()
            # print("")
            pre_move_state = state

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = pick_random_action_among_max(q[state])

            state, reward, done, info = env.step(action)

            # Update step
            q[pre_move_state][action] += alpha * (reward + gamma * max(learned_q_function[state]) - q[pre_move_state][action])
            q_max_total += max(q[pre_move_state])

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                if reward == 1.0:
                    total_times_won += 1
                else:
                    total_times_lost += 1
                all_episode_q_max.append(q_max_total / t)
                # env.render()
                break

    print("\nNumber of episodes executed:", number_of_episodes)
    print("Total times won:", total_times_won)
    print("Total times lost:", total_times_lost)
    print("Average reward per episode:", float(total_times_won) / number_of_episodes)
    plot(all_episode_q_max, "Q max total", [0, 1], gamma, alpha, epsilon)


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


def print_q_table(q_table):
    for r in range(len(q_table)):
        print("State:", r, q_table[r])


def plot(data, label, y_range, gamma, alpha, epsilon):
    plt.plot(data, "b-", label="gamma=%f, alpha=%f, epsilon=%f"%(gamma, alpha, epsilon))
    plt.ylim(y_range)
    plt.ylabel(label)
    plt.xlabel("Episode number")

    plt.legend()
    plt.show()

q_frozen_lake()
