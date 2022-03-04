# @author Metro
# @time 2021/10/14
import torch.cuda
import torch.nn.functional as F


class Config(object):
    """ Object to hold the config requirements for an agent. """

    def __init__(self):
        self.seed = None
        self.train = True
        self.evaluate = True
        self.evaluate_internal = 5
        self.environment = 'FreewheelingIntersection_v0'
        self.num_episodes_to_run = None
        self.file_to_save = None
        self.hyperparameters = None
        self.env_parameters = None
        self.standard_deviation_results = 1.0
        self.randomise_random_seed = True
        self.save_freq = 5
        self.simulations_num = 10
        self.rolling_score_window = 5
        self.runs_per_agent = 10
        self.use_GPU = True
        self.agent_name = 'P-DQN'
        self.demand = None
        self.ceil = True
        self.env_parameters = {
            'cells': 32,
            'lane_length_high': 240.,
            'speed_high': 100.,
            'edge_ids': ['north_in', 'east_in', 'south_in', 'west_in'],
            'vehicles_types': ['NW_right', 'NS_through', 'NE_left',
                               'EN_right', 'EW_through', 'ES_left',
                               'SE_right', 'SN_through', 'SW_left',
                               'WS_right', 'WE_through', 'WN_left'],
            'yellow': 3,
            'simulation_steps': 3600,
            'n_steps': 5,
            'alpha': 0.2,  # TODO
        }

        self.hyperparameters = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'epsilon_initial': 0.3,
            'epsilon_final': 0.01,
            'epsilon_decay': 5000,
            'replay_memory_size': 1e6,
            'batch_size': 64,
            'gamma': 0.99,
            'lr_critic': 1e-5,
            'lr_actor': 1e-4,
            'lr_alpha': 1e-2,
            'tau_actor': 0.01,
            'tau_critic': 0.01,
            'critic_hidden_layers': (256, 128, 64),
            'actor_hidden_layers': (256, 128, 64),
            'random_pick_steps': 10000,
            'updates_per_step': 2,
            'maximum_episodes': 2000,
            'alpha': 0.2,
        }

        self.agent_to_color_dictionary = {
            'P-DQN': '#0000FF',
            'intelligent_light': '#0E0E0F',
        }
