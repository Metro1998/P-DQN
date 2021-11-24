# @author Metro
# @time 2021/10/14


class config(object):
    """ Object to hold the config requirements for an agent. """

    def __init__(self):
        self.seed = None
        self.environment = None
        self.requirements_to_solve_game = None
        self.num_episodes_to_run = None
        self.file_to_save_data_results = None
        self.file_to_save_results_graph = None
        self.runs_per_agent = None
        self.visualise_overall_results = None
        self.visualise_individual_results = None
        self.hyperparameters = None
        self.env_parameters = None
        self.use_GPU = None
        self.overwrite_existing_results_file = None
        self.save_model = False
        self.standard_deviation_results = 1.0
        self.randomise_random_seed = True
        self.show_solution_score = False
        self.debug_mode = False

        self.env_parameters = {
            'PHASE_NUM': 8,
            'ACTION_LOW': 5.,
            'ACTION_HIGH': 20.,
            'PAD_LENGTH': 25.,
            'LANE_LENGTH_HIGH': 250.,
            'SPEED_HIGH': 100.,
            'EDGE_IDS': ['north_in', 'east_in', 'south_in', 'west_in'],
            'VEHICLES_TYPES': ['NW_right', 'NS_through', 'NE_left',
                               'EN_right', 'EW_through', 'ES_left',
                               'SE_right', 'SN_through', 'SW_left',
                               'WS_right', 'WE_through', 'WN_left'],
            'YELLOW': 3,
            'LANE_LENGTH': 234.12,
            'SIMULATION_STEPS': 3600,
            'N_STEPS': 5,
            'ALPHA': 0.2,  # TODO
        }
