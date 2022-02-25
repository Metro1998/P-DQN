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
            'phase_num': 8,
            'action_low': 5.,
            'action_high': 20.,
            'cells': 32,
            'lane_length_high': 250.,
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
            'epsilon_initial': 1.0,
            'epsilon_final': 0.05,
            'epsilon_decay': 5000,
            'replay_memory_size': 1e6,
            'initial_memory_threshold': 0,
            'batch_size': 64,
            'gamma': 0.99,
            'learning_rate_QNet': 1e-4,
            'learning_rate_ParamNet': 1e-5,
            'clip_grad': 10,
            'loss_func': F.smooth_l1_loss,
            'tau_actor': 0.01,
            'tau_actor_param': 0.01,
            'adv_hidden_layers': (256, 128, 64),
            'val_hidden_layers': (256, 128, 64),
            'param_hidden_layers': (256, 128, 64),
            'random_pick_steps': 10000,
            'updates_per_step': 1,
            'maximum_episodes': 500,
        }

        self.agent_to_color_dictionary = {
            'P-DQN': '#0000FF',
            'intelligent_light': '#0E0E0F',
        }
