# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import warnings

warnings.filterwarnings("ignore")

import conf
import numpy as np
from obp.ope import RegressionModel
from plots import show_loss, show_value
from policy import gen_eps_greedy
from policylearners import (
    GradientBasedPolicyLearner,
    OPFVPolicyLearner,
    Prognosticator,
    RegBasedPolicyLearner,
    normalize_time_by_t_oldest_and_future,
)
from regression_model_time import RegressionModelTime, RegressionModelTimeStructure
from sklearn.ensemble import RandomForestRegressor
from synthetic_time import unix_time_to_time_structure_n_tree


def OPL(
    dataset,
    dataset_test,
    dataset_train,
    time_at_evaluation_start,
    time_at_evaluation_end,
    round: int = 0,
    flag_plot_loss: bool = True,
    flag_plot_value: bool = True,
    num_time_structure_for_OPFV_reward=conf.num_true_time_structure_for_OPFV_reward,
    n_actions: int = conf.n_actions,
    dim_context: int = conf.dim_context,
    max_iter: int = conf.max_iter,
    batch_size: int = conf.batch_size,
    num_time_learn: int = conf.num_time_learn,
):
    true_value_of_learned_policies = dict()
    # Derive the value V(pi_b) of the logging policy pi_b
    # by using the logging policy pi_b and
    # the true expected reward q(x, a)
    pi_0_value = (
        (dataset_test["expected_reward"] * np.squeeze(dataset_test["pi_b"]))
        .sum(1)
        .mean()
    )

    if conf.flag_include_behavior_policy == True:
        true_value_of_learned_policies["behavior"] = pi_0_value

    # Generate an evaluation policy via the epsilon-greedy rule
    action_dist_test = gen_eps_greedy(
        expected_reward=dataset_test["expected_reward"],
        is_optimal=True,
        eps=conf.eps,
    )

    # actulal policy value
    optimal_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=dataset_test["expected_reward"],
        action_dist=action_dist_test,
    )
    if conf.flag_include_best_policy == True:
        true_value_of_learned_policies["best"] = optimal_policy_value

    reg = RegBasedPolicyLearner(
        dim_x=dim_context,
        num_actions=n_actions,
        max_iter=max_iter,
        batch_size=batch_size,
    )

    # learned policy by regression based approach (N_test * |A|)
    reg.fit(dataset_train, dataset_test)
    pi_reg = reg.predict(dataset_test)

    if conf.flag_include_RegBased == True:
        # Calculate the true value of the learned policy by regresion based approach
        true_value_of_learned_policies["reg"] = (
            (dataset_test["expected_reward"] * pi_reg).sum(1).mean()
        )

    ips = GradientBasedPolicyLearner(
        dim_x=dim_context,
        num_actions=n_actions,
        max_iter=max_iter,
        batch_size=batch_size,
    )

    # learned policy by policy gradient approach with IPS (N_test * |A|)
    ips.fit(dataset_train, dataset_test)
    pi_ips = ips.predict(dataset_test)

    if conf.flag_include_IPS_PG == True:
        # Calculate the true value of the learned policy by policy gradient approach with IPS
        true_value_of_learned_policies["ips-pg"] = (
            (dataset_test["expected_reward"] * pi_ips).sum(1).mean()
        )

    # Machine evaluation model to estimate the reward function (:math:`q(x,a):= \mathbb{E}[r|x,a]`).
    reg_model = RegressionModel(
        # Number of actions.
        n_actions=dataset.n_actions,
        # Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
        action_context=dataset_train["action_context"],
        # A machine evaluation model used to estimate the reward function.
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=conf.random_state + round
        ),
    )

    # Fit the regression model on given logged bandit data and estimate the expected rewards on the same data.
    # Returns
    #  q_hat: array-like, shape (n_rounds, n_actions, len_list)
    #  Expected rewards of new data estimated by the regression model.
    estimated_rewards = reg_model.fit_predict(
        context=dataset_train["context"],  # context; x
        action=dataset_train["action"],  # action; a
        reward=dataset_train["reward"],  # reward; r
        # Number of folds in the cross-fitting procedure.
        n_folds=2,
        random_state=conf.random_state + round,
    )
    estimated_rewards = np.squeeze(estimated_rewards, axis=2)

    dr = GradientBasedPolicyLearner(
        dim_x=dim_context,
        num_actions=n_actions,
        max_iter=max_iter,
        batch_size=batch_size,
    )

    # # Calculate the estimated expected reward \hat{q}(x, a)

    # learned policy by policy gradient approach with DR (N_test * |A|)
    dr.fit(dataset_train, dataset_test, q_hat=estimated_rewards)
    pi_dr = dr.predict(dataset_test)

    if conf.flag_include_DR_PG == True:
        true_value_of_learned_policies["dr-pg"] = (
            (dataset_test["expected_reward"] * pi_dr).sum(1).mean()
        )

    if conf.flag_include_Prognosticator == True:
        true_value_candidate_list_for_Prognosticator = []
        for (
            num_features_for_Prognosticator
        ) in conf.num_features_for_Prognosticator_list:
            for i in range(len(conf.phi_scalar_func_list)):
                prog = Prognosticator(
                    dim_x=dim_context,
                    num_actions=n_actions,
                    max_iter=max_iter,
                    batch_size=batch_size,
                    true_num_time_structures=conf.num_time_structure_for_logged_data,
                    num_features_for_Prognosticator=conf.num_time_structure_for_logged_data,
                    time_feature_func_for_Prognosticator_scalar=conf.phi_scalar_func_list[
                        i
                    ],
                    time_feature_func_for_Prognosticator_vec=conf.phi_vector_func_list[
                        i
                    ],
                    num_parameters=num_features_for_Prognosticator,
                    t_oldest=conf.t_oldest,
                    t_now=conf.t_now,
                    time_at_evaluation_start=time_at_evaluation_start,
                    time_at_evaluation_end=time_at_evaluation_end,
                    num_time_learn=num_time_learn,
                )

                prog.fit(dataset_train, dataset_test)
                pi_prog = prog.predict(dataset_test)

                true_value_candidate_for_Prognosticator = (
                    (dataset_test["expected_reward"] * pi_prog).sum(1).mean()
                )
                true_value_candidate_list_for_Prognosticator.append(
                    true_value_candidate_for_Prognosticator
                )

        true_value_of_learned_policies["prognosticator"] = max(
            true_value_candidate_list_for_Prognosticator
        )

    # Machine evaluation model to estimate the reward function (:math:`q(x,a):= \mathbb{E}[r|x,a]`).
    reg_model_time = RegressionModelTime(
        # Number of actions.
        n_actions=dataset.n_actions,
        # Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
        action_context=dataset_train["action_context"],
        # A machine evaluation model used to estimate the reward function.
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=conf.random_state + round
        ),
    )
    # Fit the regression model on given logged bandit data and estimate the expected rewards on the same data.
    # Returns
    #  q_hat: array-like, shape (n_rounds, n_actions, len_list)
    #  Expected rewards of new data estimated by the regression model.
    estimated_rewards_time = reg_model_time.fit_predict(
        context=dataset_train["context"],  # context; x
        time=normalize_time_by_t_oldest_and_future(dataset_train["time"]),  # time: t
        action=dataset_train["action"],  # action; a
        reward=dataset_train["reward"],  # reward; r
        n_folds=2,
        random_state=conf.random_state + round,
    )
    estimated_rewards_time = np.squeeze(estimated_rewards_time, axis=2)

    # Machine evaluation model to estimate the reward function (:math:`q(x,a):= \mathbb{E}[r|x,a]`).
    reg_model_time_structure = RegressionModelTimeStructure(
        # Number of actions.
        n_actions=dataset.n_actions,
        # Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
        action_context=dataset_train["action_context"],
        # A machine evaluation model used to estimate the reward function.
        base_model=RandomForestRegressor(
            n_estimators=10, max_samples=0.8, random_state=conf.random_state + round
        ),
    )

    def phi_scalar_func_for_OPFV(unix_time):
        return unix_time_to_time_structure_n_tree(
            unix_time, num_time_structure_for_OPFV_reward
        )

    # Vectorize the phi function
    phi_vector_func = np.vectorize(phi_scalar_func_for_OPFV)
    # (\phi(t_1), \cdots, \phi(t_n)) Time structure of the time we observed
    time_structure = phi_vector_func(dataset_train["time"])

    # Fit the regression model on given logged bandit data and estimate the expected rewards on the same data.
    # Returns
    #  g_hat: array-like, shape (n_rounds, n_actions, len_list)
    #  Expected rewards of new data estimated by the regression model.
    hat_g_x_phi_t_a = reg_model_time_structure.fit_predict(
        context=dataset_train["context"],  # context; x
        time_structure=time_structure,  # time structure: phi(t)
        action=dataset_train["action"],  # action; a
        reward=dataset_train["reward"],  # reward; r
        n_folds=2,
        random_state=conf.random_state + round,
    )

    opfv_opl = OPFVPolicyLearner(
        dim_x=dim_context,
        num_actions=n_actions,
        batch_size=batch_size,
        max_iter=max_iter,
        phi_scalar_func_for_OPFV=phi_scalar_func_for_OPFV,
        num_time_structure_for_OPFV_reward=num_time_structure_for_OPFV_reward,
        time_at_evaluation_start=time_at_evaluation_start,
        time_at_evaluation_end=time_at_evaluation_end,
        reg_model_time=None,
        reg_model_time_struture=reg_model_time_structure,
        num_time_learn=num_time_learn,
    )

    # # Calculate the estimated expected reward \hat{q}(x, a)

    opfv_opl.fit(
        dataset=dataset_train, dataset_test=dataset_test, q_hat=hat_g_x_phi_t_a
    )

    pi_opfv = opfv_opl.predict(dataset_test)

    true_value_of_learned_policies["opfv"] = (
        (dataset_test["expected_reward"] * pi_opfv).sum(1).mean()
    )

    if flag_plot_loss == True:
        show_loss(opfv_opl, ips, dr, reg)
    if flag_plot_value == True:
        show_value(opfv_opl, ips, dr, reg)

    return true_value_of_learned_policies, pi_0_value
