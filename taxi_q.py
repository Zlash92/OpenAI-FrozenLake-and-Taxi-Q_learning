from __future__ import print_function
import gym
import random
import matplotlib.pyplot as plt


def q_taxi(gamma=0.99, alpha=0.1, epsilon=0.01):
    """

    :param gamma: discount factor
    :param alpha: learning rate
    :return:
    """
    env = gym.make('Taxi-v1')
    number_of_states = 500

    all_episode_q_total = []  # Accumulated q values for each episode
    all_total_rewards = []  # Accumulated rewards for each episode
    all_episode_iteration_counts = []
    total_times_won = 0

    number_of_episodes = 5000
    iterations_per_episode = 1000

    # Index-> direction 0->left, 1->down, 2->right, 3->up
    q = {}
    for i in range(number_of_states):
        q[i] = [0, 0, 0, 0, 0, 0]

    for episode_count in range(number_of_episodes):
        print("\nStarting episode:", episode_count)
        state = env.reset()  # Starts episode in state 0
        q_total = 0.0
        total_reward = 0.0
        for t in range(iterations_per_episode):
            # print("Episode:", episode_count, "Time step:", t)
            # env.render()
            # print("")
            pre_move_state = state

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = pick_random_action_among_max(q[state])

            # q_total += q[pre_move_state][action]
            state, reward, done, info = env.step(action)

            # Update step
            q[pre_move_state][action] += alpha * (reward + gamma * max(q[state]) - q[pre_move_state][action])
            q_total += max(q[pre_move_state])
            total_reward += reward

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                # print("Passenger dropped of at destination")
                total_times_won += 1
                # env.render()
                all_episode_q_total.append(q_total/t)
                all_total_rewards.append(total_reward)
                all_episode_iteration_counts.append(t)
                break

    print("\nNumber of episodes executed:", number_of_episodes)
    print("Total times successfully dropped off passenger:", total_times_won)
    print("Average number of iterations before solution is found:", sum(all_episode_iteration_counts) / len(all_episode_iteration_counts))
    # plot(all_episode_q_total, "Total q-accumulation", [0, 1000], gamma, alpha, epsilon)
    plot(all_total_rewards, "Total reward", [-1000, 200], gamma, alpha, epsilon)
    plot(all_episode_iteration_counts, "Number of iterations before solution", [0, 500], gamma, alpha, epsilon)


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

q_taxi()