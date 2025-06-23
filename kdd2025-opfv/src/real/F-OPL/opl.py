# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import warnings

warnings.filterwarnings("ignore")

import time

import conf
import numpy as np
from obp.ope import RegressionModel
from plots import show_loss, show_value_test, show_value_train
from policylearners import (
    GradientBasedPolicyLearner,
    OPFVPolicyLearner,
    Prognosticator,
    RegBasedPolicyLearner,
)
from regression_model_time import RegressionModelTimeStructure
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

NUM_DAY_OF_WEEK = 7


def calcualte_DM(pi_learned, dataset_test, n_test):
    ### DM evaluation ###
    element_wise_value_DM = (dataset_test["expected_reward"] * pi_learned).sum(1)
    value_DM = element_wise_value_DM.mean()
    return value_DM


def calcualte_IPS(pi_learned, dataset_test, n_test):
    ### IPS evaluation ###
    pscore_pi_learned = pi_learned[np.arange(n_test), dataset_test["action"]]
    pscore_behavior = dataset_test["pscore"]
    iw = pscore_pi_learned / pscore_behavior

    element_wise_value_IPS = iw * dataset_test["reward"]
    value_IPS = element_wise_value_IPS.mean()
    return value_IPS


def calcualte_SNIPS(pi_learned, dataset_test, n_test):
    ### IPS evaluation ###
    pscore_pi_learned = pi_learned[np.arange(n_test), dataset_test["action"]]
    pscore_behavior = dataset_test["pscore"]
    iw = pscore_pi_learned / pscore_behavior
    normalizing_factor = iw.mean()

    element_wise_value_SNIPS = iw / normalizing_factor * dataset_test["reward"]
    value_SNIPS = element_wise_value_SNIPS.mean()
    return value_SNIPS


def calcualte_SNDR(pi_learned, dataset_test, n_test):
    ### IPS evaluation ###
    pscore_pi_learned = pi_learned[np.arange(n_test), dataset_test["action"]]
    pscore_behavior = dataset_test["pscore"]
    iw = pscore_pi_learned / pscore_behavior
    normalizing_factor = iw.mean()

    expected_reward_factual = dataset_test["expected_reward"][
        np.arange(n_test), dataset_test["action"]
    ]
    element_wise_value_SNDR = (dataset_test["expected_reward"] * pi_learned).sum(1) + (
        iw / normalizing_factor
    ) * (dataset_test["reward"] - expected_reward_factual)
    value_SNDR = element_wise_value_SNDR.mean()
    return value_SNDR


def OPL(
    dataset,
    dataset_test,
    dataset_train,
    time_test,
    round: int = 0,
    flag_plot_loss: bool = True,
    flag_plot_value_test: bool = True,
    flag_plot_value_train: bool = True,
    num_time_structure_for_OPFV_reward=conf.num_time_structure_for_OPFV_reward,
    phi_scalar_func_for_OPFV=conf.phi_scalar_func_for_OPFV,
    n_actions: int = None,
    dim_context: int = conf.dim_context,
    max_iter: int = conf.max_iter,
    batch_size: int = conf.batch_size,
    num_time_learn: int = conf.num_time_learn,
    pi_learned_list_all_results: list = None,
):
    if conf.flag_show_block_OPL == True:
        print(f"\n#################### START of OPL ####################")

    n_test = dataset_test["n_rounds"]

    ##############################################################################################
    ## RegBased
    ##############################################################################################

    if conf.flag_include_RegBased == True:
        time_start_reg_based = time.time()

        reg = RegBasedPolicyLearner(
            dim_x=dim_context,
            num_actions=n_actions,
            max_iter=max_iter,
            batch_size=batch_size,
            evaluation_rate_init=conf.learning_rate,
            solver=conf.solver,
        )

        # learned policy by regression based approach (N_test * |A|)
        reg.fit(dataset_train, dataset_test)
        pi_reg = reg.predict(dataset_test)
        print(f"pi_reg = {pi_reg}")

        time_end_reg_based = time.time()

        execution_time_reg_based = time_end_reg_based - time_start_reg_based
        print(
            f"Regression-based exectution time = {execution_time_reg_based / 60:.3f} mins"
        )

    ##############################################################################################
    ## IPS-PG
    ##############################################################################################

    if conf.flag_include_IPS_PG == True:
        time_start_ips = time.time()

        ips = GradientBasedPolicyLearner(
            dim_x=dim_context,
            num_actions=n_actions,
            max_iter=max_iter,
            batch_size=batch_size,
            evaluation_rate_init=conf.learning_rate,
            solver=conf.solver,
        )

        # learned policy by policy gradient approach with IPS (N_test * |A|)
        ips.fit(dataset_train, dataset_test)
        pi_ips = ips.predict(dataset_test)

        time_end_ips = time.time()
        execution_time_ips = time_end_ips - time_start_ips
        print(f"IPS exectution time = {execution_time_ips / 60:.3f} mins")

    ##############################################################################################
    ## DR-PG
    ##############################################################################################

    if conf.flag_include_DR_PG == True:
        time_start_dr = time.time()

        time_start_regression_model_fit = time.time()
        # Machine evaluation model to estimate the reward function (:math:`q(x,a):= \mathbb{E}[r|x,a]`).
        reg_model = RegressionModel(
            # Number of actions.
            n_actions=n_actions,
            # Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
            action_context=dataset["action_context"],
            # A machine evaluation model used to estimate the reward function.
            base_model=RandomForestRegressor(
                n_estimators=10,
                max_samples=0.8,
                random_state=conf.random_state + round,
                n_jobs=-1,
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
            n_folds=3,
            random_state=conf.random_state + round,
        )
        estimated_rewards = np.squeeze(estimated_rewards, axis=2)

        time_end_regression_model_fit = time.time()
        execution_time_regression_model_fit = (
            time_end_regression_model_fit - time_start_regression_model_fit
        )
        print(
            f"RegressionModel fitting and predition time = {execution_time_regression_model_fit / 60:.3f} mins"
        )

        dr = GradientBasedPolicyLearner(
            dim_x=dim_context,
            num_actions=n_actions,
            max_iter=max_iter,
            batch_size=batch_size,
            evaluation_rate_init=conf.learning_rate,
            solver=conf.solver,
        )

        # learned policy by policy gradient approach with DR (N_test * |A|)
        dr.fit(dataset_train, dataset_test, q_hat=estimated_rewards)
        pi_dr = dr.predict(dataset_test)

        time_end_dr = time.time()
        execution_time_dr = time_end_dr - time_start_dr
        print(f"DR exectution time = {execution_time_dr / 60:.3f} mins")

    ##############################################################################################
    ## Prognosticator
    ##############################################################################################

    if conf.flag_include_Prognosticator == True:
        time_start_prognosticator = time.time()
        true_value_candidate_list_for_Prognosticator_DM = []
        true_value_candidate_list_for_Prognosticator_IPS = []
        true_value_candidate_list_for_Prognosticator_SNIPS = []
        true_value_candidate_list_for_Prognosticator_SNDR = []
        pi_prog_candidate_list = []
        prog_candidate_list = []
        for (
            num_features_for_Prognosticator
        ) in conf.num_features_for_Prognosticator_list:
            for i in tqdm(
                range(len(conf.phi_scalar_func_list)),
                desc=f"num_features_for_Prognosticator = {num_features_for_Prognosticator}",
            ):
                prog = Prognosticator(
                    dim_x=dim_context,
                    num_actions=n_actions,
                    max_iter=max_iter,
                    batch_size=batch_size,
                    num_features_for_Prognosticator=conf.num_time_structure_for_logged_data,
                    num_time_structures_in_a_week=conf.num_time_structures_in_a_week,
                    time_structure_for_episode_division=conf.phi_scalar_func_for_Prognosticator,
                    time_feature_func_for_Prognosticator_scalar=conf.phi_scalar_func_list[
                        i
                    ],
                    time_feature_func_for_Prognosticator_vec=conf.phi_vector_func_list[
                        i
                    ],
                    num_parameters=num_features_for_Prognosticator,
                    t_oldest=dataset["t_oldest"],
                    t_now=dataset["t_now"],
                    t_future=dataset["t_future"],
                    time_test=time_test,
                    num_time_learn=num_time_learn,
                    evaluation_rate_init=conf.learning_rate,
                    solver=conf.solver,
                )

                prog.fit(dataset_train, dataset_test)

                pi_prog = prog.predict(dataset_test)
                pi_prog_candidate_list.append(pi_prog)
                prog_candidate_list.append(prog)

                n_test = dataset_test["n_rounds"]

                # Evaluate the learned policy
                true_value_candidate_for_Prognosticator_DM = calcualte_DM(
                    pi_prog, dataset_test, n_test
                )
                true_value_candidate_for_Prognosticator_IPS = calcualte_IPS(
                    pi_prog, dataset_test, n_test
                )
                true_value_candidate_for_Prognosticator_SNIPS = calcualte_SNIPS(
                    pi_prog, dataset_test, n_test
                )
                true_value_candidate_for_Prognosticator_SNDR = calcualte_SNDR(
                    pi_prog, dataset_test, n_test
                )

                # Append the estimated true value of the learned policy
                true_value_candidate_list_for_Prognosticator_DM.append(
                    true_value_candidate_for_Prognosticator_DM
                )
                true_value_candidate_list_for_Prognosticator_IPS.append(
                    true_value_candidate_for_Prognosticator_IPS
                )
                true_value_candidate_list_for_Prognosticator_SNIPS.append(
                    true_value_candidate_for_Prognosticator_SNIPS
                )
                true_value_candidate_list_for_Prognosticator_SNDR.append(
                    true_value_candidate_for_Prognosticator_SNDR
                )

        index_max_value_DM = true_value_candidate_list_for_Prognosticator_DM.index(
            max(true_value_candidate_list_for_Prognosticator_DM)
        )
        index_max_value_IPS = true_value_candidate_list_for_Prognosticator_IPS.index(
            max(true_value_candidate_list_for_Prognosticator_IPS)
        )
        index_max_value_SNIPS = (
            true_value_candidate_list_for_Prognosticator_SNIPS.index(
                max(true_value_candidate_list_for_Prognosticator_SNIPS)
            )
        )
        index_max_value_SNDR = true_value_candidate_list_for_Prognosticator_SNDR.index(
            max(true_value_candidate_list_for_Prognosticator_SNDR)
        )

        pi_prognosticator_DM = pi_prog_candidate_list[index_max_value_DM]
        pi_prognosticator_IPS = pi_prog_candidate_list[index_max_value_IPS]
        pi_prognosticator_SNIPS = pi_prog_candidate_list[index_max_value_SNIPS]
        pi_prognosticator_SNDR = pi_prog_candidate_list[index_max_value_SNDR]
        prog_DM = prog_candidate_list[index_max_value_DM]
        prog_IPS = prog_candidate_list[index_max_value_IPS]
        prog_SNIPS = prog_candidate_list[index_max_value_SNIPS]
        prog_SNDR = prog_candidate_list[index_max_value_SNDR]

    time_end_prognosticator = time.time()
    execution_time_prognosticator = time_end_prognosticator - time_start_prognosticator
    print(
        f"Prognosticator exectution time = {execution_time_prognosticator / 60:.3f} mins"
    )

    ##############################################################################################
    ## OPFV-PG
    ##############################################################################################

    time_start_opfv = time.time()
    time_start_regression_model_time_fit = time.time()

    # Machine evaluation model to estimate the reward function (:math:`q(x,a):= \mathbb{E}[r|x,a]`).
    reg_model_time_structure = RegressionModelTimeStructure(
        # Number of actions.
        n_actions=n_actions,
        # Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
        action_context=dataset["action_context"],
        # A machine evaluation model used to estimate the reward function.
        base_model=RandomForestRegressor(
            n_estimators=10,
            max_samples=0.8,
            random_state=conf.random_state + round,
            n_jobs=-1,
        ),
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
        n_folds=3,
        random_state=conf.random_state + round,
    )

    time_end_regression_model_time_fit = time.time()
    execution_time_regression_model_time_fit = (
        time_end_regression_model_time_fit - time_start_regression_model_time_fit
    )
    print(
        f"RegressionModelTimeStructure fitting and predition time = {execution_time_regression_model_time_fit / 60:.3f} mins"
    )

    opfv_opl = OPFVPolicyLearner(
        dim_x=dim_context,
        num_actions=n_actions,
        batch_size=batch_size,
        max_iter=max_iter,
        phi_scalar_func_for_OPFV=phi_scalar_func_for_OPFV,
        num_time_structure=num_time_structure_for_OPFV_reward,
        time_test=time_test,
        reg_model_time=None,
        reg_model_time_struture=reg_model_time_structure,
        num_time_learn=num_time_learn,
        evaluation_rate_init=conf.learning_rate,
        solver=conf.solver,
        flag_tune=False,
        phi_scalar_func_for_OPFV_list=None,
    )

    opfv_opl.fit(
        dataset=dataset_train, dataset_test=dataset_test, q_hat=hat_g_x_phi_t_a
    )

    pi_opfv = opfv_opl.predict(dataset_test)

    time_end_opfv = time.time()
    execution_time_opfv = time_end_opfv - time_start_opfv
    print(f"OPFV exectution time = {execution_time_opfv / 60:.3f} mins")

    # ##############################################################################################
    # ## OPFV-PG tuned
    # ##############################################################################################
    # time_start_opfv_tuned = time.time()

    # opfv_tuned_opl = OPFVPolicyLearner(
    #     dim_x=dim_context,
    #     num_actions=n_actions,
    #     batch_size=batch_size,
    #     max_iter=max_iter,
    #     phi_scalar_func_for_OPFV=phi_scalar_func_for_OPFV,
    #     num_time_structure = num_time_structure_for_OPFV_reward,
    #     time_test = time_test,
    #     reg_model_time = None,
    #     reg_model_time_struture = reg_model_time_structure,
    #     num_time_learn = num_time_learn,
    #     evaluation_rate_init=conf.learning_rate,
    #     solver=conf.solver,
    #     flag_tune = True,
    #     phi_scalar_func_for_OPFV_list = conf.phi_scalar_func_for_OPFV_list,
    #     )

    # opfv_tuned_opl.fit(dataset=dataset_train,
    #             dataset_test=dataset_test,
    #             q_hat=hat_g_x_phi_t_a)

    # pi_opfv_tuned = opfv_tuned_opl.predict(dataset_test)

    # time_end_opfv_tuned = time.time()
    # execution_time_opfv_tuned = time_end_opfv_tuned - time_start_opfv_tuned
    # print(f'OPFV (tuned) exectution time = {execution_time_opfv_tuned / 60:.3f} mins')

    if flag_plot_loss == True:
        show_loss(opfv_opl, ips, dr, reg)
    if flag_plot_value_test == True:
        show_value_test(opfv_opl, ips, dr, reg, prog_DM)
    if flag_plot_value_train == True:
        show_value_train(opfv_opl, ips, dr, reg, prog_DM)

    pi_learned_list = {}

    pi_learned_list["pi_reg"] = pi_reg
    pi_learned_list["pi_ips"] = pi_ips
    pi_learned_list["pi_dr"] = pi_dr
    pi_learned_list["pi_prognosticator_DM"] = pi_prognosticator_DM
    pi_learned_list["pi_prognosticator_IPS"] = pi_prognosticator_IPS
    pi_learned_list["pi_prognosticator_SNIPS"] = pi_prognosticator_SNIPS
    pi_learned_list["pi_prognosticator_SNDR"] = pi_prognosticator_SNDR
    pi_learned_list["pi_opfv"] = pi_opfv
    # pi_learned_list["pi_opfv_tuned"] = pi_opfv_tuned
    pi_learned_list_all_results.append(pi_learned_list)
    # return pi_reg, pi_ips, pi_dr, pi_prognosticator_DM, pi_prognosticator_IPS, pi_prognosticator_SNIPS, pi_prognosticator_SNDR, pi_opfv, pi_opfv_tuned
    return (
        pi_reg,
        pi_ips,
        pi_dr,
        pi_prognosticator_DM,
        pi_prognosticator_IPS,
        pi_prognosticator_SNIPS,
        pi_prognosticator_SNDR,
        pi_opfv,
    )


def OPL_OPFV_tune_phi(
    dataset,
    dataset_test,
    dataset_train,
    time_test,
    round: int = 0,
    num_time_structure_for_OPFV_reward=conf.num_time_structure_for_OPFV_reward,
    phi_scalar_func_for_OPFV=conf.phi_scalar_func_for_OPFV,
    n_actions: int = None,
    dim_context: int = conf.dim_context,
    max_iter: int = conf.max_iter,
    batch_size: int = conf.batch_size,
    num_time_learn: int = conf.num_time_learn,
):
    if conf.flag_show_block_OPL == True:
        print(f"\n#################### START of OPL ####################")

    ##############################################################################################
    ## OPFV-PG tuned
    ##############################################################################################

    time_start_regression_model_time_fit = time.time()

    # Machine evaluation model to estimate the reward function (:math:`q(x,a):= \mathbb{E}[r|x,a]`).
    reg_model_time_structure = RegressionModelTimeStructure(
        # Number of actions.
        n_actions=n_actions,
        # Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
        action_context=dataset["action_context"],
        # A machine evaluation model used to estimate the reward function.
        base_model=RandomForestRegressor(
            n_estimators=10,
            max_samples=0.8,
            random_state=conf.random_state + round,
            n_jobs=-1,
        ),
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
        n_folds=3,
        random_state=conf.random_state + round,
    )

    time_end_regression_model_time_fit = time.time()
    execution_time_regression_model_time_fit = (
        time_end_regression_model_time_fit - time_start_regression_model_time_fit
    )
    print(
        f"RegressionModelTimeStructure fitting and predition time = {execution_time_regression_model_time_fit / 60:.3f} mins"
    )

    time_start_opfv_tuned = time.time()

    opfv_tuned_opl = OPFVPolicyLearner(
        dim_x=dim_context,
        num_actions=n_actions,
        batch_size=batch_size,
        max_iter=max_iter,
        phi_scalar_func_for_OPFV=phi_scalar_func_for_OPFV,
        num_time_structure=num_time_structure_for_OPFV_reward,
        time_test=time_test,
        reg_model_time=None,
        reg_model_time_struture=reg_model_time_structure,
        num_time_learn=num_time_learn,
        evaluation_rate_init=conf.learning_rate,
        solver=conf.solver,
        flag_tune=True,
        phi_scalar_func_for_OPFV_list=conf.phi_scalar_func_for_OPFV_list,
    )

    opfv_tuned_opl.fit(
        dataset=dataset_train, dataset_test=dataset_test, q_hat=hat_g_x_phi_t_a
    )

    pi_opfv_tuned = opfv_tuned_opl.predict(dataset_test)

    time_end_opfv_tuned = time.time()
    execution_time_opfv_tuned = time_end_opfv_tuned - time_start_opfv_tuned
    print(f"OPFV (tuned) exectution time = {execution_time_opfv_tuned / 60:.3f} mins")

    return pi_opfv_tuned


def calculate_DM_IPS_SNIPS_SNDR_values(
    pi_learned,
    dataset_test,
    n_test,
    key_value_of_dic,
    true_value_of_learned_policies_DM,
    true_value_of_learned_policies_IPS,
    true_value_of_learned_policies_SNIPS,
    true_value_of_learned_policies_SNDR,
):
    true_value_of_learned_policies_DM[key_value_of_dic] = calcualte_DM(
        pi_learned, dataset_test, n_test
    )
    true_value_of_learned_policies_IPS[key_value_of_dic] = calcualte_IPS(
        pi_learned, dataset_test, n_test
    )
    true_value_of_learned_policies_SNIPS[key_value_of_dic] = calcualte_SNIPS(
        pi_learned, dataset_test, n_test
    )
    true_value_of_learned_policies_SNDR[key_value_of_dic] = calcualte_SNDR(
        pi_learned, dataset_test, n_test
    )


# Evaluate the learned policy by OPL algorithm using DM and IPS methods
def evaluate_OPL_algorithm(
    dataset_test,
    pi_reg=None,
    pi_ips=None,
    pi_dr=None,
    pi_prognosticator_DM=None,
    pi_prognosticator_IPS=None,
    pi_prognosticator_SNIPS=None,
    pi_prognosticator_SNDR=None,
    pi_opfv=None,
    pi_opfv_tuned=None,
    test_policy_value_list_DM_all_results=None,
    test_policy_value_list_IPS_all_results=None,
    test_policy_value_list_SNIPS_all_results=None,
    test_policy_value_list_SNDR_all_results=None,
    round=None,
):
    n_test = dataset_test["n_rounds"]

    true_value_of_learned_policies_DM = dict()
    true_value_of_learned_policies_IPS = dict()
    true_value_of_learned_policies_SNIPS = dict()
    true_value_of_learned_policies_SNDR = dict()

    pi_behavior = np.squeeze(dataset_test["pi_b"])

    # Behavior Policy Evaluation
    if conf.flag_include_behavior_policy:
        calculate_DM_IPS_SNIPS_SNDR_values(
            pi_learned=pi_behavior,
            dataset_test=dataset_test,
            n_test=n_test,
            key_value_of_dic="pi_b",
            true_value_of_learned_policies_DM=true_value_of_learned_policies_DM,
            true_value_of_learned_policies_IPS=true_value_of_learned_policies_IPS,
            true_value_of_learned_policies_SNIPS=true_value_of_learned_policies_SNIPS,
            true_value_of_learned_policies_SNDR=true_value_of_learned_policies_SNDR,
        )

    # Regression-based method evaluation
    if conf.flag_include_RegBased == True and not isinstance(pi_reg, type(None)):
        calculate_DM_IPS_SNIPS_SNDR_values(
            pi_learned=pi_reg,
            dataset_test=dataset_test,
            n_test=n_test,
            key_value_of_dic="reg",
            true_value_of_learned_policies_DM=true_value_of_learned_policies_DM,
            true_value_of_learned_policies_IPS=true_value_of_learned_policies_IPS,
            true_value_of_learned_policies_SNIPS=true_value_of_learned_policies_SNIPS,
            true_value_of_learned_policies_SNDR=true_value_of_learned_policies_SNDR,
        )

    # IPS-PG evaluation
    if conf.flag_include_IPS_PG == True and not isinstance(pi_ips, type(None)):
        calculate_DM_IPS_SNIPS_SNDR_values(
            pi_learned=pi_ips,
            dataset_test=dataset_test,
            n_test=n_test,
            key_value_of_dic="ips-pg",
            true_value_of_learned_policies_DM=true_value_of_learned_policies_DM,
            true_value_of_learned_policies_IPS=true_value_of_learned_policies_IPS,
            true_value_of_learned_policies_SNIPS=true_value_of_learned_policies_SNIPS,
            true_value_of_learned_policies_SNDR=true_value_of_learned_policies_SNDR,
        )

    # DR-PG evaluation
    if conf.flag_include_DR_PG == True and not isinstance(pi_dr, type(None)):
        calculate_DM_IPS_SNIPS_SNDR_values(
            pi_learned=pi_dr,
            dataset_test=dataset_test,
            n_test=n_test,
            key_value_of_dic="dr-pg",
            true_value_of_learned_policies_DM=true_value_of_learned_policies_DM,
            true_value_of_learned_policies_IPS=true_value_of_learned_policies_IPS,
            true_value_of_learned_policies_SNIPS=true_value_of_learned_policies_SNIPS,
            true_value_of_learned_policies_SNDR=true_value_of_learned_policies_SNDR,
        )

    # Prognosticator evaluation
    if not isinstance(pi_prognosticator_DM, type(None)):
        true_value_of_learned_policies_DM["prognosticator"] = calcualte_DM(
            pi_learned=pi_prognosticator_DM, dataset_test=dataset_test, n_test=n_test
        )
        true_value_of_learned_policies_IPS["prognosticator"] = calcualte_IPS(
            pi_learned=pi_prognosticator_IPS, dataset_test=dataset_test, n_test=n_test
        )
        true_value_of_learned_policies_SNIPS["prognosticator"] = calcualte_SNIPS(
            pi_learned=pi_prognosticator_SNIPS, dataset_test=dataset_test, n_test=n_test
        )
        true_value_of_learned_policies_SNDR["prognosticator"] = calcualte_SNDR(
            pi_learned=pi_prognosticator_SNDR, dataset_test=dataset_test, n_test=n_test
        )

    # OPFV-PG evaluation
    if not isinstance(pi_opfv, type(None)):
        calculate_DM_IPS_SNIPS_SNDR_values(
            pi_learned=pi_opfv,
            dataset_test=dataset_test,
            n_test=n_test,
            key_value_of_dic="opfv",
            true_value_of_learned_policies_DM=true_value_of_learned_policies_DM,
            true_value_of_learned_policies_IPS=true_value_of_learned_policies_IPS,
            true_value_of_learned_policies_SNIPS=true_value_of_learned_policies_SNIPS,
            true_value_of_learned_policies_SNDR=true_value_of_learned_policies_SNDR,
        )

    # OPFV-PG (tuneed) evaluation
    if not isinstance(pi_opfv_tuned, type(None)):
        calculate_DM_IPS_SNIPS_SNDR_values(
            pi_learned=pi_opfv_tuned,
            dataset_test=dataset_test,
            n_test=n_test,
            key_value_of_dic="opfv (tuned)",
            true_value_of_learned_policies_DM=true_value_of_learned_policies_DM,
            true_value_of_learned_policies_IPS=true_value_of_learned_policies_IPS,
            true_value_of_learned_policies_SNIPS=true_value_of_learned_policies_SNIPS,
            true_value_of_learned_policies_SNDR=true_value_of_learned_policies_SNDR,
        )

    if conf.flag_show_block_OPL == True:
        print(f"#################### END of OPL ####################\n")
    ### Show the obtained result if needed ###
    if conf.flag_show_round_level_result == True:
        print(
            f"ROUND {round + 1}/{conf.n_seeds}: test_policy_value_list_DM = {true_value_of_learned_policies_DM}"
        )
        print(
            f"ROUND {round + 1}/{conf.n_seeds}: test_policy_value_list_IPS = {true_value_of_learned_policies_IPS}"
        )
        print(
            f"ROUND {round + 1}/{conf.n_seeds}: test_policy_value_list_SNIPS = {true_value_of_learned_policies_SNIPS}"
        )
        print(
            f"ROUND {round + 1}/{conf.n_seeds}: test_policy_value_list_SNDR = {true_value_of_learned_policies_SNDR}"
        )

    ### Append the obtained result ###
    test_policy_value_list_DM_all_results.append(true_value_of_learned_policies_DM)
    test_policy_value_list_IPS_all_results.append(true_value_of_learned_policies_IPS)
    test_policy_value_list_SNIPS_all_results.append(
        true_value_of_learned_policies_SNIPS
    )
    test_policy_value_list_SNDR_all_results.append(true_value_of_learned_policies_SNDR)

    return (
        true_value_of_learned_policies_DM,
        true_value_of_learned_policies_IPS,
        true_value_of_learned_policies_SNIPS,
        true_value_of_learned_policies_SNDR,
    )
