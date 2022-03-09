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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_ornstein_noise = True
        # the simulation_step in Env() is fixed (1800 in default), and this term makes sense in route_generator.
        # when you update simulation_step don't forget that in Env()
        self.simulation_step = 1800

        self.hyperparameters = {
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
            'initial_memory_threshold': 256,
            'alpha': 0.2,
            'loss_func': F.smooth_l1_loss,
            'clip_grad': 10,
            'init_std': 0.1,

        }

        self.others = {
            'indexed': True,
            'weighted': True,
            'average': True,
            'random_weighted': True,
            'inverting_gradients': True,
            'zero_index_gradients': True,
        }

        self.agent_to_color_dictionary = {
            'P-DQN': '#0000FF',
            'intelligent_light': '#0E0E0F',
        }
