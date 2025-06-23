# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


from typing import Callable

import numpy as np


# fourier function (scalar version)
def fourier_scalar(x, n, K_plus_delta):
    array = np.array(1)
    for i in range(1, n + 1):
        array = np.append(array, np.sin(2 * np.pi * i * x / (K_plus_delta)))
        array = np.append(array, np.cos(2 * np.pi * i * x / (K_plus_delta)))
    return array


# fourier function (vector version)
def fourier_vec(x, n, K_plus_delta):
    matrix = np.ones(len(x)).astype(int)
    for i in range(1, n + 1):
        matrix = np.column_stack((matrix, np.sin(2 * np.pi * i * x / (K_plus_delta))))
        matrix = np.column_stack((matrix, np.cos(2 * np.pi * i * x / (K_plus_delta))))
    return matrix


# exponetial function (scalar version)
def exponential_scalar(x, n, K_plus_delta):
    array = np.array(1)
    for i in range(1, n + 1):
        array = np.append(array, np.exp(-i * x / (K_plus_delta)))
        array = np.append(array, -np.exp(-i * x / (K_plus_delta)))
    return array


# exponetial function (vector version)
def exponential_vec(x, n, K_plus_delta):
    matrix = np.ones(len(x)).astype(int)
    for i in range(1, n + 1):
        matrix = np.column_stack((matrix, np.exp(-i * x / (K_plus_delta))))
        matrix = np.column_stack((matrix, -np.exp(i * x / (K_plus_delta))))
    return matrix


def create_array(N, K):
    quotient = N // K
    remainder = N % K

    if remainder == 0:
        A = np.array([quotient] * K)
    else:
        A = np.array([quotient + 1] * remainder + [quotient] * (K - remainder))

    return A


def create_episode_context(horizon_array, N, K):
    episode_context = np.zeros((N, K))
    count = 0
    for i in range(len(horizon_array)):
        for j in range(horizon_array[i]):
            episode_context[j + count, i] = 1
        count += horizon_array[i]
    return episode_context


def Prognosticator(
    num_episodes: int,
    phi_scalar_func,
    phi_vector_func,
    reward: np.ndarray,
    action: np.ndarray,
    pscore: np.ndarray,
    action_dist: np.ndarray,
    num_episodes_after_logged_data: int = 1,
    num_features_for_Prognosticator: int = 3,
) -> np.ndarray:
    # Obtain the numbe of episodes (K) and the number of rounds (n)
    K = num_episodes
    n_rounds = action.shape[0]

    # Create the vector that contains independent variables when we use linear regression X \in R^K
    X = np.arange(1, K + 1)
    # Map each element of X using given feature function to create Phi \in R^(K \times d)
    Phi = phi_vector_func(X, num_features_for_Prognosticator)

    # Create the vector that contains importance weight w(x_i, t_i, a_i) for each round i
    iw = action_dist[np.arange(n_rounds), action] / pscore
    round_rewards = reward * iw

    # How to choose each horizon
    horizon_array = create_array(n_rounds, K)
    episode_context = create_episode_context(horizon_array, n_rounds, K)

    # Calculate the vector that contains independent variables V_k(\pi_e; D) Y \in R^K
    Y = round_rewards @ episode_context / horizon_array

    # Calculate the estimate V_t(\pi_e; D) by using OLS estimator
    episode_at_evaluation = K + num_episodes_after_logged_data
    if phi_vector_func == exponential_vec:
        episode_at_evaluation = episode_at_evaluation / K
    try:
        # If there exists the inverse matrix, then use it
        V_t = (
            phi_scalar_func(episode_at_evaluation, num_features_for_Prognosticator)
            @ np.linalg.inv(Phi.T @ Phi)
            @ Phi.T
            @ Y
        )

    except np.linalg.LinAlgError:
        # If the matrix is singular, then we use pseudo inverse matrix instead of inverse matirx
        V_t = (
            phi_scalar_func(episode_at_evaluation, num_features_for_Prognosticator)
            @ np.linalg.pinv(Phi.T @ Phi)
            @ Phi.T
            @ Y
        )
    return V_t


########################################################
### OPFV general case robust to non-stationary context
########################################################
def OPFV(
    time_at_eval: int,
    estimated_rewards_by_reg_model: np.ndarray,
    estimated_rewards_by_reg_model_at_eval: np.ndarray,
    reward: np.ndarray,
    action: np.ndarray,
    time: np.ndarray,
    pscore: np.ndarray,
    action_dist: np.ndarray,
    action_dist_at_eval: np.ndarray,
    phi_scalar_func: Callable = None,  # \phi_r(t)
    phi_scalar_func_for_context: Callable = None,  # \phi_x(t)
    flag_robust_to_non_stationary_context: bool = False,  # if OPFV is robust to non-stationary context or not
    flag_use_true_P_phi_t_for_reward: bool = False,
    P_phi_t_true_for_reward: float = None,  # P(\phi_r(t))
    flag_use_true_P_phi_t_for_context: bool = False,
    P_phi_t_true_for_context: float = None,  # P(\phi_x(t))
    flag_use_true_P_phi_t_for_context_reward: bool = False,
    P_phi_t_true_for_context_reward: float = None,  # P(\phi_{x, r}(t))
) -> np.ndarray:
    # The number of observation: n_rounds
    n = action.shape[0]

    ############ About \phi_r(t) ############
    # Vectorize the phi function
    phi_vector_func_for_reward = np.vectorize(phi_scalar_func)
    # (\phi_r(t_1), \cdots, \phi_r(t_n)) Time structure of the time we observed
    time_structure_for_reward = phi_vector_func_for_reward(time)
    # \phi(t_e) Time structure of the time at the evaluation
    time_structure_at_eval_for_reward = phi_vector_func_for_reward(time_at_eval)
    # 1 if \ind{\phi_r(t_i) = \phi_r(t_e)}, and 0 otherwiese
    indicator_phi_r = (
        time_structure_for_reward == time_structure_at_eval_for_reward
    ).astype(int)

    if flag_use_true_P_phi_t_for_reward == False:
        # \hat{P}(\phi(t_e)) Estimated probability of observing the time structure identical to the one of the time at the evaluation
        est_P_phi_t_for_reward = indicator_phi_r.mean()
    else:
        est_P_phi_t_for_reward = P_phi_t_true_for_reward

    indicator_phi_x = np.zeros(n)
    indicator_phi_x_r = np.zeros(n)

    est_P_phi_t_for_context = None
    est_P_phi_t_for_context_reward = None

    if flag_robust_to_non_stationary_context == True:
        ############ About \phi_x(t) ############
        # Vectorize the phi function
        phi_vector_func_for_context = np.vectorize(phi_scalar_func_for_context)
        # (\phi_r(t_1), \cdots, \phi_r(t_n)) Time structure of the time we observed
        time_structure_for_rcontext = phi_vector_func_for_context(time)
        # \phi(t_e) Time structure of the time at the evaluation
        time_structure_at_eval_for_context = phi_vector_func_for_context(time_at_eval)
        # 1 if \ind{\phi_r(t_i) = \phi_r(t_e)}, and 0 otherwiese
        indicator_phi_x = (
            time_structure_for_rcontext == time_structure_at_eval_for_context
        ).astype(int)

        if flag_use_true_P_phi_t_for_context == False:
            # \hat{P}(\phi(t_e)) Estimated probability of observing the time structure identical to the one of the time at the evaluation
            est_P_phi_t_for_context = indicator_phi_x.mean()
        else:
            est_P_phi_t_for_context = P_phi_t_true_for_context

        ############ About \phi_{x, r}(t) ############
        indicator_phi_x_r = indicator_phi_x * indicator_phi_r

        if flag_use_true_P_phi_t_for_context_reward == False:
            # \hat{P}(\phi(t_e)) Estimated probability of observing the time structure identical to the one of the time at the evaluation
            est_P_phi_t_for_context_reward = indicator_phi_x_r.mean()
        else:
            est_P_phi_t_for_context_reward = P_phi_t_true_for_context_reward

    # (w(x_1, t_1, a_1), \cdots, w(x_n, t_n, a_n)) importance wieght for each round
    iw = action_dist_at_eval[np.arange(n), action] / pscore

    # (\hat{f}(x_1, t_1, a_1), \cdots, \hat{f}(x_n, t_n, a_n))
    f_hat_factual = estimated_rewards_by_reg_model[np.arange(n), action]

    # E_{\pi_e(a|x_i, t)}[\hat{f}(x_i, t, a)]
    estimated_rewards = np.average(
        estimated_rewards_by_reg_model_at_eval,
        weights=action_dist_at_eval,
        axis=1,
    )

    ########### OPFV only robust to non-stationary reward ###########
    if flag_robust_to_non_stationary_context == False:
        # OPFV calculate expected reward for each round
        estimate_round_rewards = (indicator_phi_r / est_P_phi_t_for_reward) * iw * (
            reward - f_hat_factual
        ) + estimated_rewards

    else:
        ########### OPFV robust to both non-stationary context and reward ###########
        estimate_round_rewards = (
            indicator_phi_x_r / est_P_phi_t_for_context_reward
        ) * iw * (reward - f_hat_factual) + (
            indicator_phi_x / est_P_phi_t_for_context
        ) * estimated_rewards

    return estimate_round_rewards
