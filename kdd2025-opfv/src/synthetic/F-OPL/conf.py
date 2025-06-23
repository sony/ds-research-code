# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import datetime

from estimators_time import fourier_scalar, fourier_vec
from synthetic_time import (
    obtain_num_time_structure,
    unix_time_to_day_of_week,
)

################### START main hyperparameters ###################
max_iter = 25
batch_size = 32
num_time_learn = 50
n_seeds = 20
num_train = 8000
num_train_list = [1000, 2000, 4000, 8000]
lambda_ratio_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
num_time_at_evaluation = 8
candidate_num_time_structure_list = range(2, 9, 1)
candidate_num_time_structure_list_for_OPFV = range(2, 17, 2)
reward_std = 1
num_test = 10000
alpha_ratio_list = lambda_ratio_list
alpha_ratio_and_lambda_ratio_list = lambda_ratio_list
################### END main hyperparameters ###################


################### START time related hyperparameters ###################
num_overlaps = 1
t_now = int(
    datetime.datetime.timestamp(
        datetime.datetime(year=2022, month=12, day=31, hour=23, minute=59, second=59)
    )
)
t_oldest = int(
    datetime.datetime.timestamp(
        datetime.datetime(
            year=2023 - num_overlaps, month=1, day=1, hour=0, minute=0, second=0
        )
    )
)
time_at_evaluation_start = int(
    datetime.datetime.timestamp(
        datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0, second=0)
    )
)
t_future = int(
    datetime.datetime.timestamp(
        datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0)
    )
)
num_time_structure_for_logged_data = 8
num_cycles_in_evaluation_period = 1
################### END time related hyperparameters ###################


################### START time structure, OPFV, Prognosticator related hyperparameters ###################
num_true_time_structure_for_OPFV_reward = num_time_structure_for_logged_data
num_episodes_for_Prognosticator = num_time_structure_for_logged_data * num_overlaps
phi_scalar_func_list = [
    fourier_scalar,
]
phi_vector_func_list = [
    fourier_vec,
]
num_features_for_Prognosticator = 3
num_features_for_Prognosticator_list = range(3, 8, 2)
flag_Prognosticator_optimality = True
################### END time structure, OPFV, Prognosticator related hyperparameters ###################


################### START non stationary context hyperparameters ###################
sample_non_stationary_context = False
time_structure_func_for_context = unix_time_to_day_of_week
num_time_structure_for_context = obtain_num_time_structure(unix_time_to_day_of_week)
p_1_coef = 3
p_2_coef = 1
################### END non stationary context hyperparameters ###################


################### START data class hyperparameters ###################
beta_list = [-3, -1, 0, 1, 3]
n_actions_list = [5, 10, 20, 40]
dim_context_list = [5, 10, 20, 40]
num_time_learn_list = [10, 20, 40, 80]
n_users_list = [50, 100, 200, 400]
beta = 0.1
alpha_ratio = 0.5
lambda_ratio = 0.5
n_actions = 10
dim_context = 10
n_users = None
flag_simple_reward = True
g_coef = 3
h_coef = 1
random_state = 12345
eps = 0
################### END data class hyperparameters ###################


################### START estimator flag hyperparameters ###################
flag_Prognosticator_with_multiple_feature_func = False
flag_include_DM = False
flag_calculate_data_driven_OPFV = True
flag_plot_loss = False
flag_plot_value = False
flag_include_behavior_policy = False
flag_include_best_policy = False
flag_include_RegBased = True
flag_include_IPS_PG = True
flag_include_DR_PG = True
flag_include_Prognosticator = True
################### END estimator flag hyperparameters ###################

markersize = 12
