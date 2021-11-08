# @author Metro
# @time 2021/11/8

"""
  Mainly based on https://github.com/cycraig/MP-DQN/blob/master/run_goal_pdqn.py
"""

import os
import click
import time
import numpy as np
import gym
import gym_goal
# TODO
from agents.pdqn import PDQNAgent

@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=20000, help='Number of episodes.', type=int)
@click.option('--evaluation_episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch_size', default=128, help='Mini batch size.', type=int)
@click.option('--gamma', default=0.95, help='Discount factor.', type=float)
@click.option('--inverting_gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
# TODO 了解inverting gradients 与 squashing function
@click.option('--initial_memory_threshold', default=128, help='Number of transitions required to start learning.',
              type=int)
@click.option('--replay_memory_size', default=20000, help='Replay memory transition capacity.', type=int)
@click.option('--epsilon_decay', default=5000,
              help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon_final', default=0.01, help='Final epsilon value.', type=float)
@click.option('--tau_actor', default=0.1, help='Soft target network update averaging factor.', type=float)
@click.option('--tau_actor_param', default=0.001, help='Soft target network update averaging factor.', type=float)
@click.option('--learning_rate_actor', default=0.001, help="Actor network learning rate.", type=float)
@click.option('--learning_rate_actor_param', default=0.00001, help="Critic network learning rate.", type=float)
@click.option('--clip_grad', default=1., help="Parameter gradient clipping limit.", type=float)
@click.option('--layers', default="(256,)", help='Hidden layers.', cls=ClickPythonLiteralOption)
# TODO
@click.option('--action_input_layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--save_freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save_dir', default="results/goal", help='Output directory.', type=str)
@click.option('--render_freq', default=100, help='How often to render / save frames of an episode.', type=int)
@click.option('--save_frames', default=False,
              help="Save render frames from the environment. Incompatible with visualise.", type=bool)
@click.option('--visualize', default=True, help="Render game states. Incompatible with save-frames.", type=bool)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_decay, epsilon_final, tau_actor, tau_actor_param, learning_rate_actor,
        learning_rate_actor_param, clip_grad, layers, action_input_layer, save_freq, save_dir, render_dir,
        save_frames, visualize)

    env = gym.make('Goal-v0')



