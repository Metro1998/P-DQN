import torch
import torch.nn.functional as F
from config import Config
from trainer import Train_and_Evaluate

config = Config()
config.seed = 1
config.train = True
config.evaluate = False
config.evaluate_internal = 5
config.environment = 'FreewheelingIntersection-v0'
config.num_episodes_to_run = 500
config.file_to_save = 'results/'
config.save_model = True
config.standard_deviation_results = 1.0
config.randomise_random_seed = True
config.save_freq = 5
config.simulations_num = 10
config.rolling_score_window = 5
config.runs_per_agent = 10
config.agent_name = 'P-DQN'
config.use_GPU = True
config.ceil = True
config.demand = [
    [1. / 12, 1. / 19, 1. / 18, 1. / 13, 1. / 16, 1. / 14, 1. / 22, 1. / 21, 1. / 20, 1. / 11, 1. / 16, 1. / 18],
    [1. / 12, 1. / 19, 1. / 18, 1. / 13, 1. / 16, 1. / 14, 1. / 12, 1. / 11, 1. / 10, 1. / 11, 1. / 16, 1. / 18]
]

config.env_parameters = {
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
config.hyperparameters = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epsilon_initial': 0.5,
    'epsilon_final': 0.01,
    'epsilon_decay': 10000,
    'replay_memory_size': 1e5,
    'initial_memory_threshold': 0,
    'batch_size': 256,
    'gamma': 0.99,
    'learning_rate_QNet': 1e-5,
    'learning_rate_ParamNet': 1e-4,
    'clip_grad': 10,
    'loss_func': F.smooth_l1_loss,
    'tau_actor': 0.01,
    'tau_actor_param': 0.01,
    'hidden_layers': (256, 128, 64),
    'param_hidden_layers': (256, 128, 64),
    'random_pick_steps': 200,
    'updates_per_step': 1,
    'maximum_episodes': 500,
}

if __name__ == "__main__":
    trainer = Train_and_Evaluate(config=config)
    trainer.train_agent()
