from __future__ import print_function
import gym
import math
import random


def q_frozen_lake():
    env = gym.make('FrozenLake-v0')

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
            next_action = action_selector()
            state, reward, done, info = env.step(next_action)


def action_selector():
    # Index: Left, Down, Right, Up
    action_weights = [0.5, 1.0, 0.5, 0.5]

    action_probabilities = probability_distribution(action_weights)
    return roulette_wheel_selection(action_probabilities)


def probability_distribution(list_of_weights):
    """
    Calculate relative probabilities of each action
    :param list_of_weights:
    :return:
    """
    total_weight = sum(list_of_weights)
    action_probabilities = map(lambda x: x/total_weight, list_of_weights)
    return action_probabilities


def softmax(list_of_weights):
    """
    Calculate probability distribution of actions using softmax based on weights
    :param list_of_weights:
    :return:
    """
    total_exp_weights = sum(map(lambda x: math.exp(x), list_of_weights))
    action_probabilties = map(lambda x: math.exp(x)/total_exp_weights, list_of_weights)
    return action_probabilties


def roulette_wheel_selection(list_of_probabilities):
    """
    Use roulette wheel selection for choosing an action based on the probabilites of each action
    :param list_of_probabilities:
    :return: index of chosen action
    """

    action_intervals = [sum(list_of_probabilities[:i + 1]) * 100 for i in range(len(list_of_probabilities))]
    roulette_ball = random.uniform(0, 100)
    for (action, interval_limit) in enumerate(action_intervals):
        if roulette_ball <= interval_limit:
            return action