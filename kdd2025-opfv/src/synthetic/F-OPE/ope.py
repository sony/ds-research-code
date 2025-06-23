# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


from typing import Callable

import conf
import numpy as np
from estimators_time import OPFV, Prognosticator, fourier_scalar, fourier_vec
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import OffPolicyEvaluation, RegressionModel
from policy import gen_eps_greedy
from sklearn.ensemble import RandomForestRegressor
from synthetic_time import (
    unix_time_to_time_structure_n_tree,
)
from utils import calculate_hat_f_train_and_eval


def run_ope(
    dataset,
    round,
    time_at_evaluation,
    estimated_policy_value_list,
    val_bandit_data,
    action_dist_val,
    num_episodes_for_Prognosticator,
    num_time_structure_from_t_now_to_time_at_evaluation: int,
    num_true_time_structure_for_OPFV_reward: int,
    num_true_time_structure_for_OPFV_for_context: int = None,
    eps=conf.eps,
    flag_calulate_robust_OPFV: bool = False,
    flag_Prognosticator_optimality: bool = True,
    num_features_for_Prognosticator_list: np.ndarray = np.ndarray([1, 2, 3]),
    true_policy_value: float = None,
    flag_include_DM: bool = True,
    flag_calculate_data_driven_OPFV: bool = False,
    candidate_num_time_structure_list: Callable = None,
):
    def phi_scalar_func_for_OPFV(unix_time):
        return unix_time_to_time_structure_n_tree(
            unix_time, num_true_time_structure_for_OPFV_reward
        )

    if flag_calulate_robust_OPFV == True:

        def phi_scalar_func_for_OPFV_for_context(unix_time):
            return unix_time_to_time_structure_n_tree(
                unix_time, num_true_time_structure_for_OPFV_for_context
            )

    def finest_time_structure(unix_time):
        return unix_time_to_time_structure_n_tree(
            unix_time, candidate_num_time_structure_list[-1]
        )

    #########################################################################################
    # DM, IPS, and DM
    #########################################################################################
    # Machine learning model to estimate the reward function (:math:`q(x,a):= \mathbb{E}[r|x,a]`).
    reg_model = RegressionModel(
        # Number of actions.
        n_actions=dataset.n_actions,
        # Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
        action_context=val_bandit_data["action_context"],
        # A machine learning model used to estimate the reward function.
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=12345 + round
        ),
    )

    # Fit the regression model on given logged bandit data and estimate the expected rewards on the same data.
    # Returns
    #  q_hat: array-like, shape (n_rounds, n_actions, len_list)
    #  Expected rewards of new data estimated by the regression model.
    estimated_rewards = reg_model.fit_predict(
        context=val_bandit_data["context"],  # context; x
        action=val_bandit_data["action"],  # action; a
        reward=val_bandit_data["reward"],  # reward; r
        # Number of folds in the cross-fitting procedure.
        n_folds=2,
        random_state=12345 + round,
    )

    if flag_include_DM == True:
        ope_estimators = [
            IPS(estimator_name="IPS"),
            DR(estimator_name="DR"),
            DM(estimator_name="DM"),
        ]
    else:
        ope_estimators = [
            IPS(estimator_name="IPS"),
            DR(estimator_name="DR"),
        ]

    ope = OffPolicyEvaluation(
        # observed data D
        bandit_feedback=val_bandit_data,
        # list of estimators
        ope_estimators=ope_estimators,
    )

    estimated_policy_values = ope.estimate_policy_values(
        action_dist=action_dist_val[:, :, np.newaxis],
        estimated_rewards_by_reg_model=estimated_rewards,
        pi_b=val_bandit_data["pi_b"],
    )

    #########################################################################################
    # Prognosticator
    #########################################################################################

    sorted_indices = np.argsort(val_bandit_data["time"])

    action_sorted_Prognosticator = val_bandit_data["action"][sorted_indices]
    reward_sorted_Prognosticator = val_bandit_data["reward"][sorted_indices]
    pscore_sorted_Prognosticator = val_bandit_data["pscore"][sorted_indices]
    action_dist_val_sorted_Prognosticator = action_dist_val[sorted_indices]

    if flag_Prognosticator_optimality == True:
        estimated_mse_list = []
        candidate_estimated_policy_value_list = []

        # For each time feature function
        for i in range(len(conf.phi_scalar_func_list)):
            # For each number of time features
            for num_features_for_Prognosticator in num_features_for_Prognosticator_list:
                # Calculate the estimated policy value by Prognosticator
                candidate_estimated_policy_value = Prognosticator(
                    num_episodes=num_episodes_for_Prognosticator,
                    phi_scalar_func=conf.phi_scalar_func_list[i],
                    phi_vector_func=conf.phi_vector_func_list[i],
                    reward=reward_sorted_Prognosticator,
                    action=action_sorted_Prognosticator,
                    pscore=pscore_sorted_Prognosticator,
                    action_dist=action_dist_val_sorted_Prognosticator,
                    num_episodes_after_logged_data=num_time_structure_from_t_now_to_time_at_evaluation,
                    num_features_for_Prognosticator=num_features_for_Prognosticator,
                )
                # Add the estimated policy value to the list of candidate estimated policy values
                candidate_estimated_policy_value_list.append(
                    candidate_estimated_policy_value
                )
                # Calculate the true MSE of the estimated policy value with true policy value
                estimated_mse = (
                    candidate_estimated_policy_value - true_policy_value
                ) ** 2
                # Add the true MSE of the estimated policy value to the list of true MSE of candidate estimated policy values
                estimated_mse_list.append(estimated_mse)

        # Obtain the index which minimizes the true MSE of the estimated policy value
        min_index = min(
            range(len(estimated_mse_list)), key=lambda i: estimated_mse_list[i]
        )
        # Set the estimated policy value with the lowest MSE
        estimated_policy_values["Prognosticator"] = (
            candidate_estimated_policy_value_list[min_index]
        )
    else:
        estimated_policy_values["Prognosticator"] = Prognosticator(
            num_episodes=num_episodes_for_Prognosticator,
            phi_scalar_func=fourier_scalar,
            phi_vector_func=fourier_vec,
            reward=reward_sorted_Prognosticator,
            action=action_sorted_Prognosticator,
            pscore=pscore_sorted_Prognosticator,
            action_dist=action_dist_val_sorted_Prognosticator,
            num_episodes_after_logged_data=num_time_structure_from_t_now_to_time_at_evaluation,
            num_features_for_Prognosticator=conf.num_features_for_Prognosticator,
        )

    # #########################################################################################
    # # OPFV
    # #########################################################################################

    n = val_bandit_data["action"].shape[0]
    time_at_eval_vec = np.full(n, time_at_evaluation)

    hat_f_x_t_a, hat_f_x_t_a_at_eval = calculate_hat_f_train_and_eval(
        phi_scalar_func_for_OPFV, val_bandit_data, dataset, time_at_eval_vec, round
    )

    _, _, hat_f_x_t_a_at_eval_true = dataset.synthesize_expected_reward(
        contexts=val_bandit_data["context"], times=time_at_eval_vec
    )

    ## make decisions on validation data
    action_dist_val_at_eval = gen_eps_greedy(
        expected_reward=hat_f_x_t_a_at_eval_true,
        is_optimal=True,
        eps=eps,
    )

    # this OPFV is OPFV with q and estimated P(\phi_r(t))
    estimated_policy_values["OPFV"] = OPFV(
        phi_scalar_func=phi_scalar_func_for_OPFV,  # \phi_r(t)
        phi_scalar_func_for_context=None,  # \phi_x(t)
        time_at_eval=time_at_evaluation,
        estimated_rewards_by_reg_model=hat_f_x_t_a,
        estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
        reward=val_bandit_data["reward"],
        action=val_bandit_data["action"],
        time=val_bandit_data["time"],
        pscore=val_bandit_data["pscore"],
        action_dist=action_dist_val,
        action_dist_at_eval=action_dist_val_at_eval,
        flag_robust_to_non_stationary_context=False,  # if OPFV is robust to non-stationary context or not
        flag_use_true_P_phi_t_for_reward=False,
        P_phi_t_true_for_reward=None,  # P(\phi_r(t))
        flag_use_true_P_phi_t_for_context=False,
        P_phi_t_true_for_context=None,  # P(\phi_x(t))
        flag_use_true_P_phi_t_for_context_reward=False,
        P_phi_t_true_for_context_reward=None,  # P(\phi_{x, r}(t))
    ).mean()

    if flag_calulate_robust_OPFV == True:
        # this OPFV is OPFV with q and estimated P(\phi_r(t))
        estimated_policy_values["robust OPFV"] = OPFV(
            phi_scalar_func=phi_scalar_func_for_OPFV,  # \phi_r(t)
            phi_scalar_func_for_context=phi_scalar_func_for_OPFV_for_context,  # \phi_x(t)
            time_at_eval=time_at_evaluation,
            estimated_rewards_by_reg_model=hat_f_x_t_a,
            estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
            reward=val_bandit_data["reward"],
            action=val_bandit_data["action"],
            time=val_bandit_data["time"],
            pscore=val_bandit_data["pscore"],
            action_dist=action_dist_val,
            action_dist_at_eval=action_dist_val_at_eval,
            flag_robust_to_non_stationary_context=True,  # if OPFV is robust to non-stationary context or not
            flag_use_true_P_phi_t_for_reward=False,
            P_phi_t_true_for_reward=None,  # P(\phi_r(t))
            flag_use_true_P_phi_t_for_context=False,
            P_phi_t_true_for_context=None,  # P(\phi_x(t))
            flag_use_true_P_phi_t_for_context_reward=False,
            P_phi_t_true_for_context_reward=None,  # P(\phi_{x, r}(t))
        ).mean()

    if flag_calculate_data_driven_OPFV == True:
        hat_f_x_t_a, hat_f_x_t_a_at_eval = calculate_hat_f_train_and_eval(
            finest_time_structure, val_bandit_data, dataset, time_at_eval_vec, round
        )

        estimated_value_with_finest_time_structure = OPFV(
            phi_scalar_func=finest_time_structure,  # \phi_r(t)
            phi_scalar_func_for_context=None,  # \phi_x(t)
            time_at_eval=time_at_evaluation,
            estimated_rewards_by_reg_model=hat_f_x_t_a,
            estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
            reward=val_bandit_data["reward"],
            action=val_bandit_data["action"],
            time=val_bandit_data["time"],
            pscore=val_bandit_data["pscore"],
            action_dist=action_dist_val,
            action_dist_at_eval=action_dist_val_at_eval,
            flag_robust_to_non_stationary_context=False,  # if OPFV is robust to non-stationary context or not
            flag_use_true_P_phi_t_for_reward=False,
            P_phi_t_true_for_reward=None,  # P(\phi_r(t))
            flag_use_true_P_phi_t_for_context=False,
            P_phi_t_true_for_context=None,  # P(\phi_x(t))
            flag_use_true_P_phi_t_for_context_reward=False,
            P_phi_t_true_for_context_reward=None,  # P(\phi_{x, r}(t))
        ).mean()
        estimated_mse_list = []
        candidate_estimated_value_list = []
        for candidate_num_time_structure in candidate_num_time_structure_list:

            def candidate_phi_scalar_func(unix_time):
                return unix_time_to_time_structure_n_tree(
                    unix_time, candidate_num_time_structure
                )

            hat_f_x_t_a, hat_f_x_t_a_at_eval = calculate_hat_f_train_and_eval(
                candidate_phi_scalar_func,
                val_bandit_data,
                dataset,
                time_at_eval_vec,
                round,
            )
            candidate_value_round_rewards = OPFV(
                phi_scalar_func=candidate_phi_scalar_func,  # \phi_r(t)
                phi_scalar_func_for_context=None,  # \phi_x(t)
                time_at_eval=time_at_evaluation,
                estimated_rewards_by_reg_model=hat_f_x_t_a,
                estimated_rewards_by_reg_model_at_eval=hat_f_x_t_a_at_eval,
                reward=val_bandit_data["reward"],
                action=val_bandit_data["action"],
                time=val_bandit_data["time"],
                pscore=val_bandit_data["pscore"],
                action_dist=action_dist_val,
                action_dist_at_eval=action_dist_val_at_eval,
                flag_robust_to_non_stationary_context=False,  # if OPFV is robust to non-stationary context or not
                flag_use_true_P_phi_t_for_reward=False,
                P_phi_t_true_for_reward=None,  # P(\phi_r(t))
                flag_use_true_P_phi_t_for_context=False,
                P_phi_t_true_for_context=None,  # P(\phi_x(t))
                flag_use_true_P_phi_t_for_context_reward=False,
                P_phi_t_true_for_context_reward=None,  # P(\phi_{x, r}(t))
            )
            candidate_estimated_value_list.append(candidate_value_round_rewards.mean())

            est_squared_bias = (
                candidate_value_round_rewards.mean()
                - estimated_value_with_finest_time_structure
            ) ** 2
            est_var = np.var(candidate_value_round_rewards, ddof=1) / len(
                candidate_value_round_rewards
            )
            estimated_mse_list.append(est_squared_bias + est_var)

        min_index = min(
            range(len(estimated_mse_list)), key=lambda i: estimated_mse_list[i]
        )

        estimated_policy_values["data-driven OPFV"] = candidate_estimated_value_list[
            min_index
        ]

    estimated_policy_values["V_t"] = true_policy_value

    estimated_policy_value_list.append(estimated_policy_values)
