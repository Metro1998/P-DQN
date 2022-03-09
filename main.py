import torch
from config import Config
from trainer import Train_and_Evaluate

config = Config()
config.seed = 123456
config.train = True
config.evaluate = False
config.evaluate_internal = 5
config.environment = 'FreewheelingIntersection-v1'
config.file_to_save = 'results/'
config.save_model = True
config.randomise_random_seed = True
config.save_freq = 5
config.rolling_score_window = 5
config.runs_per_agent = 3
config.agent_name = 'P-DQN'
config.use_GPU = True
config.ceil = True
config.updates_per_step = 1
config.maximum_episodes = 400
config.demand = [
    [1. / 22, 1. / 20, 1. / 21, 1. / 18, 1. / 16, 1. / 14, 1. / 13, 1. / 21, 1. / 20, 1. / 21, 1. / 19, 1. / 18],
    [1. / 20, 1. / 21, 1. / 18, 1. / 13, 1. / 16, 1. / 12, 1. / 12, 1. / 19, 1. / 13, 1. / 11, 1. / 16, 1. / 18]
]

config.env_parameters = {
    'cells': 32,
    'lane_length_high': 240.,
    'speed_high': 100.,
    'edge_ids': ['north_in', 'east_in', 'south_in', 'west_in'],
    'vehicles_types': ['NW_right', 'NS_through', 'NE_left',
                       'EN_right', 'EW_through', 'ES_left',
                       'SE_right', 'SN_through', 'SW_left',
                       'WS_right', 'WE_through', 'WN_left'],
    'yellow': 3,
    'simulation_steps': 7200,
}
config.hyperparameters = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epsilon_initial': 0.5,
    'epsilon_final': 0,
    'epsilon_decay': 1000,
    'replay_memory_size': 1e5,
    'batch_size': 256,
    'gamma': 0.99,
    'lr_critic': 1e-4,
    'lr_actor': 1e-3,
    'lr_alpha': 1e-2,
    'tau_critic': 0.01,
    'tau_actor': 0.01,
    'critic_hidden_layers': (256, 128, 64),
    'actor_hidden_layers': (128, 64),
    'alpha': 0.2,
}

if __name__ == "__main__":
    trainer = Train_and_Evaluate(config=config)
    trainer.train_agent()
