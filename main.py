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
config.num_episodes_to_run = 100
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


config.env_parameters = {
    'phase_num': 8,
    'action_low': 5.,
    'action_high': 20.,
    'pad_length': 25,
    'lane_length_high': 250.,
    'speed_high': 100.,
    'edge_ids': ['north_in', 'east_in', 'south_in', 'west_in'],
    'vehicles_types': ['NW_right', 'NS_through', 'NE_left',
                       'EN_right', 'EW_through', 'ES_left',
                       'SE_right', 'SN_through', 'SW_left',
                       'WS_right', 'WE_through', 'WN_left'],
    'yellow': 3,
    'lane_length': 234.12,
    'simulation_steps': 3600,
    'n_steps': 5,
    'alpha': 0.2,  # TODO

}
config.hyperparameters = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epsilon_initial': 1.0,
    'epsilon_final': 0.05,
    'epsilon_decay': 5000,
    'replay_memory_size': 1e6,
    'initial_memory_threshold': 0,
    'batch_size': 64,
    'gamma': 0.99,
    'learning_rate_actor': 1e-4,
    'learning_rate_actor_param': 1e-5,
    'clip_grad': 10,
    'loss_func': F.smooth_l1_loss,
    'tau_actor': 0.01,
    'tau_actor_param': 0.01,
    'hidden_layer_actor': (256, 128, 64),
    'hidden_layer_actor_param': (256, 128, 64),
    'random_pick_steps': 10000,
    'updates_per_step': 1,
    'maximum_episodes': 500,
}

if __name__ == "__main__":
    trainer = Train_and_Evaluate(config=config)
    trainer.train_agent()


