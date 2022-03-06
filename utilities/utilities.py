# @author Metro
# @time 2021/11/03

import torch
import matplotlib.pyplot as plt
import numpy as np


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# -------- Plot --------
def visualize_overall_agent_results(agent_results, agent_name, show_mean_and_std_range=True,
                                    agent_to_color_dictionary=None, standard_deviation_results=1,
                                    file_path_for_pic=None):
    """
    Visualize the results for one agent.

    :param file_path_for_pic:
    :param title:
    :param standard_deviation_results:
    :param agent_to_color_dictionary:
    :param agent_results: list of lists, each
    :param agent_name:
    :param show_mean_and_std_range:
    :return:
    """
    assert isinstance(agent_results, list), 'agent_results must be a list of lists.'
    assert isinstance(agent_results[0], list), 'agent_result must be a list of lists.'
    fig, ax = plt.subplots()
    color = agent_to_color_dictionary[agent_name]
    if show_mean_and_std_range:
        mean_minus_x_std, mean_results, mean_plus_x_std = get_mean_and_standard_deviation_difference(
            agent_results, standard_deviation_results)
        x_vals = list(range(len(mean_results)))
        ax.plot(x_vals, mean_results, label=agent_name, color=color)
        ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)  # TODO
        ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
        ax.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.1, color=color)
    else:
        color_idx = 0
        colors = ['red', 'blue', 'green', 'orange', 'yellow', 'purple']
        for ix, result in enumerate(agent_results):
            x_vals = list(range(len(agent_results[0])))
            ax.plot(x_vals, result, label=agent_name + '_{}'.format(ix + 1), color=color)
            color, color_idx = get_next_color(colors, color_idx)

    ax.set_facecolor('xkcd:white')
    ax.legend(loc='upper right', shadow='Ture', facecolor='inherit')
    ax.set_title(label='Training', fontsize=15, fontweight='bold')
    ax.set_ylabel('Rolling Episode Scores')
    ax.set_xlabel('Episode Number')
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.set_xlim([0, x_vals[-1]])

    y_limits = get_y_limits(agent_results)
    ax.set_ylim(y_limits)

    plt.tight_layout()
    plt.savefig(file_path_for_pic)


def get_mean_and_standard_deviation_difference(results, standard_deviation_results):
    """
    From a list of lists of specific agent results it extracts the mean result and the mean result plus or minus
    some multiple of standard deviation.

    :param standard_deviation_results:
    :param results:
    :return:
    """

    def get_results_at_a_time_step(results_, timestep):
        results_at_a_time_step = [result[timestep] for result in results_]
        return results_at_a_time_step

    def get_std_at_a_time_step(results_, timestep):
        results_at_a_time_step = [result[timestep] for result in results_]
        return np.std(results_at_a_time_step)

    mean_results = [np.mean(get_results_at_a_time_step(results, timestep)) for timestep in range(len(results[0]))]
    mean_minus_x_std = [mean_val - standard_deviation_results * get_std_at_a_time_step(results, timestep)
                        for timestep, mean_val in enumerate(mean_results)]
    mean_plus_x_std = [mean_val + standard_deviation_results * get_std_at_a_time_step(results, timestep)
                       for timestep, mean_val in enumerate(mean_results)]
    return mean_minus_x_std, mean_results, mean_plus_x_std


def get_next_color(colors=None, color_idx=None):
    """
    Gets the next color in list self.colors. If it gets to the end then it starts from beginning.

    :return:
    """

    color_idx += 1
    if color_idx >= len(colors):
        color_idx = 0

    return colors[color_idx], color_idx


def get_y_limits(results):
    """
    Extracts the minimum and maximum seen y_vals from a set of results.

    :param results:
    :return:
    """
    min_result = float('inf')
    max_result = float('-inf')
    for result in results:
        tem_max = np.max(result)
        tem_min = np.min(result)
        if tem_max > max_result:
            max_result = tem_max
        if tem_min < min_result:
            min_result = tem_min
    y_limits = [min_result, max_result]
    return y_limits


def visualize_results_per_run(agent_results, agent_name, save_freq, file_path_for_pic):
    """

    :param file_path_for_pic:
    :param save_freq:
    :param agent_name:
    :param agent_results:
    :return:
    """
    assert isinstance(agent_results, list), 'agent_results must be a list of lists.'
    fig, ax = plt.subplots()
    ax.set_facecolor('xkcd:white')
    ax.legend(loc='upper right', shadow='Ture', facecolor='inherit')
    ax.set_title(label='Episode Scores For One Specific Run', fontsize=15, fontweight='bold')
    ax.set_ylabel('Episode Scores')
    ax.set_xlabel('Episode Number')
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    x_vals = list(range(len(agent_results)))
    ax.set_xlim([0, x_vals[-1]])
    ax.set_ylim([min(agent_results), max(agent_results)])
    ax.plot(x_vals, agent_results, label=agent_name, color='blue')
    plt.tight_layout()

    Runtime = len(agent_results)
    if Runtime % save_freq == 0:
        plt.savefig(file_path_for_pic)


def observation_wrapper(state, action_pre):
    state_st_NS = [state[0], state[4], action_pre]
    state_st_EW = [state[2], state[6], action_pre]
    state_le_NS = [state[1], state[5], action_pre]
    state_le_EW = [state[3], state[7], action_pre]
    state_sl_N = [state[0], state[1], action_pre]
    state_sl_E = [state[2], state[3], action_pre]
    state_sl_S = [state[4], state[5], action_pre]
    state_sl_W = [state[6], state[7], action_pre]

    return [state_st_NS, state_st_EW, state_le_NS, state_le_EW, state_sl_N, state_sl_E, state_sl_S, state_sl_W]


# -------- Optimizer --------
class SharedAdam(torch.optim.Adam):
    """
    Shared optimizer, the parameters in the optimizer will shared in the multiprocessors.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


# ---------- Noise ----------
class OrnsteinUhlenbeckActionNoise(object):
    """
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    Source: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/utils.py
    """

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X
