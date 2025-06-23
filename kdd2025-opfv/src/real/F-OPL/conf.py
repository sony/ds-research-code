# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


from estimators_time import fourier_scalar, fourier_vec
from utils_time import (
    obtain_num_episodes_for_Prognosticator,
    obtain_num_time_structure,
    obtain_num_time_structures_in_a_week,
    unix_time_to_AM_PM,
    unix_time_to_day_of_week,
    unix_time_to_day_of_week_and_AM_PM,
    unix_time_to_day_of_week_and_four_paritions_a_day,
    unix_time_to_day_of_week_and_hour,
    unix_time_to_four_paritions_a_day,
    unix_time_to_hour,
    unix_time_to_weekday_weekend,
    unix_time_to_weekday_weekend_and_AM_PM,
    unix_time_to_weekday_weekend_and_four_paritions_a_day,
    unix_time_to_weekday_weekend_and_hour,
)

################### START main hyperparameters ###################
max_iter = 25
batch_size = 1024
num_time_learn = 10
learning_rate = 0.005
n_seeds = 10
n_actions = 100
dim_context = 60
dim_action_context = 40
solver = "adam"
################### END main hyperparameters ###################


################### START time structure, OPFV, Prognosticator related hyperparameters ###################
phi_scalar_func_for_OPFV = unix_time_to_day_of_week
phi_scalar_func_for_OPFV_list = [
    unix_time_to_day_of_week,
    unix_time_to_weekday_weekend,
    unix_time_to_hour,
    unix_time_to_four_paritions_a_day,
    unix_time_to_AM_PM,
    unix_time_to_weekday_weekend_and_AM_PM,
    unix_time_to_weekday_weekend_and_four_paritions_a_day,
    unix_time_to_weekday_weekend_and_hour,
    unix_time_to_day_of_week_and_AM_PM,
    unix_time_to_day_of_week_and_four_paritions_a_day,
    unix_time_to_day_of_week_and_hour,
]
phi_scalar_func_for_Prognosticator = phi_scalar_func_for_OPFV
num_time_structure_for_OPFV_reward = obtain_num_time_structure(phi_scalar_func_for_OPFV)
num_true_time_structure_for_OPFV_reward = num_time_structure_for_OPFV_reward
num_time_structure_for_logged_data = num_true_time_structure_for_OPFV_reward
num_episodes_for_Prognosticator = obtain_num_episodes_for_Prognosticator(
    phi_scalar_func_for_OPFV
)
num_time_structures_in_a_week = obtain_num_time_structures_in_a_week(
    phi_scalar_func_for_OPFV
)
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
################### END non stationary context hyperparameters ###################


################### START data class hyperparameters ###################
random_state = 12345
eps = 0
################### END data class hyperparameters ###################


################### START pi_b estimation hyperparameters ###################
num_epochs_for_pi = 25
learning_rate_for_pi = 0.01
batch_size_for_pi_b = 16
alpha = 0.01
flag_print_loss = False
################### END pi_b estimation hyperparameters ###################


################### START estimator flag hyperparameters ###################
flag_sample_test = False
flag_Prognosticator_with_multiple_feature_func = False
flag_include_DM = False
flag_calculate_data_driven_OPFV = True
flag_plot_loss = False
flag_plot_value_test = False
flag_plot_value_train = False
flag_include_behavior_policy = True
flag_include_best_policy = False
flag_include_RegBased = True
flag_include_IPS_PG = True
flag_include_DR_PG = True
flag_include_Prognosticator = True
flag_show_parameters = False
flag_show_context_PCA = False
flag_show_action_context_PCA = False
flag_show_round_level_result = True
flag_show_execution_time_for_preprocess = True
flag_show_num_original_dim_context = True
flag_show_num_original_dim_action_context = True
flag_show_execution_time_for_each_iter_in_OPFV = True
flag_show_block_preprocessing = True
flag_show_block_OPL = True
flag_show_num_trains_and_tests = True
################### END estimator flag hyperparameters ###################


markersize = 12
