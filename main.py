import torch
import torch.nn.functional as F
from config import Config
from trainer import Train_and_Evaluate

config = Config()
config.seed = 123456
config.train = True
config.evaluate = False
config.evaluate_internal = 5
config.environment = 'FreewheelingIntersection-v1'
config.file_to_save = 'results/'
config.standard_deviation_results = 1.0
config.save_freq = 5
config.rolling_score_window = 5
config.runs_per_agent = 3
config.agent_name = 'P-DQN'
config.use_GPU = True
config.demand = [
    [1. / 22, 1. / 20, 1. / 21, 1. / 18, 1. / 16, 1. / 14, 1. / 13, 1. / 21, 1. / 20, 1. / 21, 1. / 19, 1. / 18],
    [1. / 20, 1. / 21, 1. / 18, 1. / 13, 1. / 16, 1. / 12, 1. / 12, 1. / 19, 1. / 13, 1. / 11, 1. / 16, 1. / 18]
]
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.use_ornstein_noise = True
config.simulation_step = 1800

config.hyperparameters = {
    'epsilon_initial': 0.1,
    'epsilon_final': 0,
    'epsilon_decay': 3000,
    'replay_memory_size': 1e5,
    'batch_size': 256,
    'gamma': 0.95,
    'lr_critic': 1e-3,
    'lr_actor': 1e-4,
    'lr_alpha': 1e-2,
    'tau_critic': 0.01,
    'tau_actor': 0.01,
    'critic_hidden_layers': (256, 128, 64),
    'actor_hidden_layers': (256, 128, 64),
    'updates_per_step': 1,
    'maximum_episodes': 400,
    'loss_func': F.smooth_l1_loss,
    'clip_grad': 5,
    'init_std': None,
}

config.others = {
    'indexed': True,
    'inverting_gradients': True,
}

config.agent_to_color_dictionary = {
    'P-DQN': '#0000FF',
    'intelligent_light': '#0E0E0F',
}

if __name__ == "__main__":
    trainer = Train_and_Evaluate(config=config)
    trainer.train_agent()
