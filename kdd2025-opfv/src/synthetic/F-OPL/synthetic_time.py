# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import datetime
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from obp.dataset.base import BaseBanditDataset
from obp.dataset.reward_type import RewardType
from obp.types import BanditFeedback
from obp.utils import check_array, sample_action_fast, softmax
from scipy.stats import truncnorm
from sklearn.utils import check_random_state, check_scalar

SECONDS_PER_DAY = 24 * 60 * 60
BIG_NUM = int(1e5)
NUM_DAY_OF_WEEK = 7


coef_func_signature = Callable[
    [np.ndarray, np.ndarray, np.random.RandomState],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]


def unix_time_to_season(unix_time):
    # Convert Unix timestamp to a datetime object
    datetime_converted = datetime.datetime.fromtimestamp(unix_time)
    month = datetime_converted.month
    # Get the season as an integer (0 = Spring, 1 = Summer, 2 = Fall, 3 = Winter)
    if 1 <= month <= 3:
        return 0
    elif 4 <= month <= 6:
        return 1
    elif 7 <= month <= 9:
        return 2
    elif 10 <= month <= 12:
        return 3


def unix_time_to_month(unix_time):
    # Convert Unix timestamp to a datetime object
    datetime_converted = datetime.datetime.fromtimestamp(unix_time)
    # Get the month as an integer (0 = Jan, 1 = Feb, ..., 11 = Dec)
    month = datetime_converted.month
    return month - 1


def unix_time_to_day_of_week(unix_time):
    # Convert Unix timestamp to a datetime object
    datetime_converted = datetime.datetime.fromtimestamp(unix_time)
    # Get the day of the week as an integer (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
    weekday = datetime_converted.weekday()
    return weekday


def unix_time_to_hour(unix_time):
    # Convert Unix timestamp to a datetime object
    datetime_converted = datetime.datetime.fromtimestamp(unix_time)
    # Get the hour as an integer (0 = 0:00~0:59, 1 = 1:00~1:59, ..., 23 = 23:00~23:59)
    hour = datetime_converted.hour
    return hour


def unix_time_to_AM_PM(unix_time):
    # Convert Unix timestamp to a datetime object
    datetime_converted = datetime.datetime.fromtimestamp(unix_time)
    # Get the hour as an integer (0 = 0:00~0:59, 1 = 1:00~1:59, ..., 23 = 23:00~23:59)
    hour = datetime_converted.hour
    # Output 0 if the time is in AM, 1 if PM
    if 0 <= hour < 12:
        return 0
    else:
        return 1


def unix_time_to_season_month(unix_time):
    return unix_time_to_season(unix_time) * 12 + unix_time_to_month(unix_time)


def unix_time_to_season_month_day_of_week(unix_time):
    return (
        unix_time_to_season(unix_time) * 84
        + unix_time_to_month(unix_time) * 7
        + unix_time_to_day_of_week(unix_time)
    )


def unix_time_to_season_month_day_of_week_AM_PM(unix_time):
    return (
        unix_time_to_season(unix_time) * 168
        + unix_time_to_month(unix_time) * 14
        + unix_time_to_day_of_week(unix_time) * 2
        + unix_time_to_AM_PM(unix_time)
    )


def days_passed_to_time_structure_tree(days_passed, num_time_structure):
    max_K_th_power_of_two = 1
    for i in range(BIG_NUM):
        max_K_th_power_of_two *= 2
        if num_time_structure < max_K_th_power_of_two:
            max_K_th_power_of_two /= 2
            max_K_th_power_of_two = int(max_K_th_power_of_two)
            break
    remainder = int(num_time_structure % max_K_th_power_of_two)
    if remainder == 0:
        ref_days = 366 / max_K_th_power_of_two
        ref_intervals = 366 / max_K_th_power_of_two
        for i in range(num_time_structure):
            if days_passed <= ref_days:
                return i
            ref_days += ref_intervals

        return num_time_structure - 1
    else:
        ref_days = 366 / (max_K_th_power_of_two * 2)
        ref_intervals = 366 / (max_K_th_power_of_two * 2)
        for i in range(num_time_structure):
            if days_passed <= ref_days:
                return i
            if i + 1 < 2 * remainder:
                ref_days += ref_intervals
            else:
                ref_days += ref_intervals * 2

        return num_time_structure - 1


def unix_time_to_time_structure_n_tree(unix_time, num_time_structure):
    # Convert the Unix timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(unix_time)

    # Get the year from the datetime object
    year = dt_object.year

    # Get the first day of the year for the given year
    first_day_of_year = datetime.datetime(year, 1, 1)

    # Calculate the number of days passed in the year
    number_of_days_passed = (dt_object - first_day_of_year).days + 1

    return days_passed_to_time_structure_tree(number_of_days_passed, num_time_structure)


def obtain_num_time_structure(time_structure_func):
    if time_structure_func == unix_time_to_season:
        return 4
    elif time_structure_func == unix_time_to_month:
        return 12
    elif time_structure_func == unix_time_to_day_of_week:
        return 7
    elif time_structure_func == unix_time_to_AM_PM:
        return 2
    elif time_structure_func == unix_time_to_hour:
        return 24
    elif time_structure_func == unix_time_to_season_month:
        return 4 * 12
    elif time_structure_func == unix_time_to_season_month_day_of_week:
        return 4 * 12 * 7
    elif time_structure_func == unix_time_to_season_month_day_of_week_AM_PM:
        return 4 * 12 * 7 * 2


def obtain_num_days_in_one_cycle(time_structure_func):
    if time_structure_func == unix_time_to_season:
        return 365
    elif time_structure_func == unix_time_to_month:
        return 365
    elif time_structure_func == unix_time_to_day_of_week:
        return 7
    elif time_structure_func == unix_time_to_AM_PM:
        return 1
    elif time_structure_func == unix_time_to_hour:
        return 1
    elif time_structure_func == unix_time_to_season_month:
        return 365
    elif time_structure_func == unix_time_to_season_month_day_of_week:
        return 365
    elif time_structure_func == unix_time_to_season_month_day_of_week_AM_PM:
        return 365
    elif time_structure_func == unix_time_to_time_structure_n_tree:
        return 365


def sample_random_uniform_coefficients(
    effective_dim_action_context: int,
    effective_dim_context: int,
    random_: np.random.RandomState,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    context_coef_ = random_.uniform(-1, 1, size=effective_dim_context)
    action_coef_ = random_.uniform(-1, 1, size=effective_dim_action_context)
    context_action_coef_ = random_.uniform(
        -1, 1, size=(effective_dim_context, effective_dim_action_context)
    )
    return context_coef_, action_coef_, context_action_coef_


# Normalize time to [0, 1] (or [0, scale])
def normalize_time(time, t_oldest, t_future, scale=1):
    return scale * (time - t_oldest) / (t_future - t_oldest)


@dataclass
class SyntheticBanditWithTimeDataset(BaseBanditDataset):
    n_actions: int
    dim_context: int = 1
    n_users: int = None

    # The oldest unix time when we can potentially observe logged bandit data
    t_oldest: int = int(
        datetime.datetime.timestamp(datetime.datetime(year=2022, month=1, day=1))
    )
    # The latest unix time when we can potentially observe logged bandit data
    t_now: int = int(
        datetime.datetime.timestamp(datetime.datetime(year=2022, month=6, day=1))
    )
    # The latest future unix time when we want to evaluate a target policy
    t_future: int = int(
        datetime.datetime.timestamp(datetime.datetime(year=2023, month=1, day=1))
    )

    num_time_structure: int = 7

    num_time_structure_for_context: int = 7

    # q(x, t, a) = \lambda * g(x, \phi(t), a) + (1 - \lambda) * h(x, t, a)
    lambda_ratio: float = 0.95

    # p(x|t) = \alpha * p_1(x|\phi_x(t)) + (1 - \alpha) * p_2(x|t)
    alpha_ratio: float = 0.95

    reward_type: str = RewardType.CONTINUOUS.value

    flag_simple_reward: bool = True

    sample_non_stationary_context: bool = False

    g_coef: int = 3
    h_coef: int = 1

    p_1_coef: int = 3
    p_2_coef: int = 1

    reward_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None

    reward_std: float = 1.0
    action_context: Optional[np.ndarray] = None

    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    beta: float = 1.0
    n_deficient_actions: int = 0
    random_state: int = 12345
    dataset_name: str = "synthetic_bandit_with_time_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.dim_context, "dim_context", int, min_val=1)
        check_scalar(self.beta, "beta", (int, float))
        check_scalar(
            self.n_deficient_actions,
            "n_deficient_actions",
            int,
            min_val=0,
            max_val=self.n_actions - 1,
        )

        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)

        if RewardType(self.reward_type) not in [
            RewardType.BINARY,
            RewardType.CONTINUOUS,
        ]:
            raise ValueError(
                f"`reward_type` must be either '{RewardType.BINARY.value}' or '{RewardType.CONTINUOUS.value}',"
                f"but {self.reward_type} is given.'"
            )
        check_scalar(self.reward_std, "reward_std", (int, float), min_val=0)
        if self.reward_function is None:
            self.expected_reward = self.sample_contextfree_expected_reward()
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            self.reward_min = 0
            self.reward_max = 1e10

        # one-hot encoding characterizing actions.
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)
        else:
            check_array(
                array=self.action_context, name="action_context", expected_dim=2
            )
            if self.action_context.shape[0] != self.n_actions:
                raise ValueError(
                    "Expected `action_context.shape[0] == n_actions`, but found it False."
                )
        self._define_param_for_q_and_h()

        def true_time_structure_func_for_reward(unix_time):
            return unix_time_to_time_structure_n_tree(
                unix_time, self.num_time_structure
            )

        self.time_structure_func = true_time_structure_func_for_reward

        def true_time_structure_func_for_context(unix_time):
            return unix_time_to_time_structure_n_tree(
                unix_time, self.num_time_structure_for_context
            )

        self.time_structure_func_for_context = true_time_structure_func_for_context

    # Set the parameters used for construting g(x, \phi(t), a) and h(x, t, a)
    def _define_param_for_q_and_h(self) -> None:
        # Sample parameters from Unif([-h_coef, h_coef]) for generating h(x, t, a)
        self.theta_x = self.random_.uniform(
            low=-self.h_coef, high=self.h_coef, size=self.dim_context
        )
        self.theta_t = self.random_.uniform(low=-self.h_coef, high=self.h_coef, size=2)
        self.theta_a = self.random_.uniform(
            low=-self.h_coef, high=self.h_coef, size=self.n_actions
        )
        self.theta_t_a_1 = self.random_.uniform(
            low=-self.h_coef, high=self.h_coef, size=self.n_actions
        )
        self.theta_t_a_2 = self.random_.uniform(
            low=-self.h_coef, high=self.h_coef, size=3
        )
        self.N_x_a = self.random_.uniform(
            low=-self.h_coef, high=self.h_coef, size=(self.dim_context, self.n_actions)
        )
        self.theta_finer_phi_t = self.random_.uniform(
            low=-self.h_coef, high=self.h_coef, size=NUM_DAY_OF_WEEK
        )
        self.N_finer_phi_t_a_1 = self.random_.uniform(
            low=-self.h_coef, high=self.h_coef, size=(NUM_DAY_OF_WEEK, self.n_actions)
        )
        self.P_x_finer_phi_t_a = self.random_.uniform(
            low=-self.h_coef,
            high=self.h_coef,
            size=(self.dim_context, NUM_DAY_OF_WEEK, self.n_actions),
        )

        # Sample parameters from Unif([-g_coef, g_coef]) for generating g(x, \phi(t), a)
        self.psi_x = self.random_.uniform(
            low=-self.g_coef, high=self.g_coef, size=self.dim_context
        )
        self.psi_phi_t = self.random_.uniform(
            low=-self.g_coef, high=self.g_coef, size=self.num_time_structure
        )
        self.psi_a = self.random_.uniform(
            low=-self.g_coef, high=self.g_coef, size=self.n_actions
        )
        self.M_phi_t_a = self.random_.uniform(
            low=-self.g_coef,
            high=self.g_coef,
            size=(self.num_time_structure, self.n_actions),
        )
        self.M_x_a = self.random_.uniform(
            low=-self.g_coef, high=self.g_coef, size=(self.dim_context, self.n_actions)
        )
        self.P_x_phi_t_a = self.random_.uniform(
            low=-self.g_coef,
            high=self.g_coef,
            size=(self.dim_context, self.num_time_structure, self.n_actions),
        )

        # Sample parameters from Unif([-p_1_corf, p_1_coef]) for generating p_1(x|t)
        self.gamma = self.random_.uniform(
            low=-self.p_1_coef,
            high=self.p_1_coef,
            size=self.num_time_structure_for_context,
        )

        # Sample parameters from Unif([-p_2_corf, p_2_coef]) for generating p_1(x|t)
        self.delta = self.random_.uniform(
            low=-self.p_2_coef, high=self.p_2_coef, size=5
        )

    @property
    def len_list(self) -> int:
        """Length of recommendation lists, slate size."""
        return 1

    def sample_contextfree_expected_reward(self) -> np.ndarray:
        """Sample expected reward for each action from the uniform distribution."""
        return self.random_.uniform(size=self.n_actions)

    def calc_expected_reward(self, context: np.ndarray) -> np.ndarray:
        """Sample expected rewards given contexts"""
        # sample reward for each round based on the reward function
        if self.reward_function is None:
            expected_reward_ = np.tile(self.expected_reward, (context.shape[0], 1))
        else:
            expected_reward_ = self.reward_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )

        return expected_reward_

    def sample_reward_given_expected_reward(
        self,
        expected_reward: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Sample reward given expected rewards"""
        expected_reward_factual = expected_reward[np.arange(action.shape[0]), action]
        if RewardType(self.reward_type) == RewardType.BINARY:
            reward = self.random_.binomial(n=1, p=expected_reward_factual)
        elif RewardType(self.reward_type) == RewardType.CONTINUOUS:
            mean = expected_reward_factual
            a = (self.reward_min - mean) / self.reward_std
            b = (self.reward_max - mean) / self.reward_std
            reward = truncnorm.rvs(
                a=a,
                b=b,
                loc=mean,
                scale=self.reward_std,
                random_state=self.random_state,
            )
        else:
            raise NotImplementedError

        return reward

    def sample_reward(self, context: np.ndarray, action: np.ndarray) -> np.ndarray:
        check_array(array=context, name="context", expected_dim=2)
        check_array(array=action, name="action", expected_dim=1)
        if context.shape[0] != action.shape[0]:
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0]`, but found it False"
            )
        if not np.issubdtype(action.dtype, np.integer):
            raise ValueError("the dtype of action must be a subdtype of int")

        expected_reward_ = self.calc_expected_reward(context)

        return self.sample_reward_given_expected_reward(expected_reward_, action)

    def synthesize_expected_reward(self, contexts, times):
        n_rounds = contexts.shape[0]

        # Convert Unix timestamp to a datetime object
        finer_time_structure_func = np.vectorize(datetime.datetime.utcfromtimestamp)
        dt_objects = finer_time_structure_func(times)

        # Assuming dt_objects is a NumPy array of datetime objects
        get_day_of_week = np.vectorize(lambda dt: dt.weekday())
        days_of_week = get_day_of_week(dt_objects)

        finer_time_structure_context = np.zeros(shape=(n_rounds, NUM_DAY_OF_WEEK))

        row_indices = np.arange(n_rounds)
        column_indices = days_of_week

        finer_time_structure_context[row_indices, column_indices] = 1

        time_structure_func_vec = np.vectorize(self.time_structure_func)

        time_structures = time_structure_func_vec(times)

        time_structure_context = np.zeros(shape=(n_rounds, self.num_time_structure))

        row_indices = np.arange(n_rounds)
        column_indices = time_structures

        time_structure_context[row_indices, column_indices] = 1

        # Synthetize h(x, t, a)
        # if h(x, t, a) is a simple or comlex function

        # Initialize h(x, t, a) by zero matrix
        h_x_t_a_ = np.zeros((n_rounds, self.n_actions))

        # Synthesize each of the componets to synthesize h(x, t, a)

        if self.dim_context == 10:
            h_1_x = (contexts[:, 0:6].sum(axis=1) < 2.5) * self.theta_x[0]
            h_1_x += (contexts[:, 7:9].sum(axis=1) < -0.5) * self.theta_x[1]
            h_1_x += (contexts[:, 2:5].sum(axis=1) > 2.0) * self.theta_x[2]
        else:
            h_1_x = contexts @ self.theta_x / self.dim_context

        h_2_t = finer_time_structure_context @ self.theta_finer_phi_t

        h_3_a = self.action_context @ self.theta_a

        if self.dim_context == 10:
            shrinked_contexts = np.concatenate(
                [
                    (contexts[:, 0:4].sum(axis=1) < 3).reshape(-1, 1),
                    (contexts[:, 2:9].sum(axis=1) > 2.5).reshape(-1, 1),
                    (contexts[:, 1:7].sum(axis=1) < 1.5).reshape(-1, 1),
                    (contexts[:, 6:10].sum(axis=1) > -1.5).reshape(-1, 1),
                ],
                axis=1,
            )
            h_5_x_a = shrinked_contexts @ self.N_x_a[0:4, :] @ self.action_context
        else:
            h_5_x_a = contexts @ self.N_x_a @ self.action_context / self.dim_context

        h_6_t_a = (
            finer_time_structure_context @ self.N_finer_phi_t_a_1 @ self.action_context
        )

        if self.dim_context == 10:
            shrinked_contexts = np.concatenate(
                [
                    (contexts[:, 0:4].sum(axis=1) < 4).reshape(-1, 1),
                    (contexts[:, 2:9].sum(axis=1) > 3.5).reshape(-1, 1),
                    (contexts[:, 2:5].sum(axis=1) > 1.5).reshape(-1, 1),
                    (contexts[:, 5:10].sum(axis=1) < -2.5).reshape(-1, 1),
                ],
                axis=1,
            )
            h_7_x_phi_t_a = np.einsum(
                "ij,jkl->ikl", shrinked_contexts, self.P_x_finer_phi_t_a[0:4, :, :]
            )
            h_7_x_phi_t_a = np.einsum(
                "ijk,ij->ik", h_7_x_phi_t_a, finer_time_structure_context
            )
            h_7_x_phi_t_a = h_7_x_phi_t_a @ self.action_context
        else:
            h_7_x_phi_t_a = (
                np.einsum("ij,jkl->ikl", contexts, self.P_x_finer_phi_t_a)
                / self.dim_context
            )
            h_7_x_phi_t_a = np.einsum(
                "ijk,ij->ik", h_7_x_phi_t_a, finer_time_structure_context
            )
            h_7_x_phi_t_a = h_7_x_phi_t_a @ self.action_context

        h_x_t_a_ = (
            h_1_x[:, np.newaxis]
            + h_2_t[:, np.newaxis]
            + h_3_a
            + h_5_x_a
            + h_6_t_a
            + h_7_x_phi_t_a
        )

        # Synthetize g(x, \phi(t), a)
        g_x_phi_t_a_ = np.zeros((n_rounds, self.n_actions))

        if self.dim_context == 10:
            g_1_x = (contexts[:, 0:4].sum(axis=1) < 1.5) * self.psi_x[0]
            g_1_x += (contexts[:, 5:9].sum(axis=1) < -0.5) * self.psi_x[1]
            g_1_x += (contexts[:, 3:5].sum(axis=1) > 3.0) * self.psi_x[2]
            g_1_x += (contexts[:, 6:10].sum(axis=1) < 1.0) * self.psi_x[3]
        else:
            g_1_x = contexts @ self.psi_x / self.dim_context

        g_2_phi_t = time_structure_context @ self.psi_phi_t

        g_6_phi_t_a = time_structure_context @ self.M_phi_t_a @ self.action_context

        if self.dim_context == 10:
            shrinked_contexts = np.concatenate(
                [
                    (contexts[:, 0:4].sum(axis=1) < 4).reshape(-1, 1),
                    (contexts[:, 5:9].sum(axis=1) > 3).reshape(-1, 1),
                    (contexts[:, 2:10].sum(axis=1) < -2.5).reshape(-1, 1),
                ],
                axis=1,
            )
            g_7_x_phi_t_a = np.einsum(
                "ij,jkl->ikl", shrinked_contexts, self.P_x_phi_t_a[0:3, :, :]
            )
            g_7_x_phi_t_a = np.einsum(
                "ijk,ij->ik", g_7_x_phi_t_a, time_structure_context
            )
            g_7_x_phi_t_a = g_7_x_phi_t_a @ self.action_context
        else:
            g_7_x_phi_t_a = (
                np.einsum("ij,jkl->ikl", contexts, self.P_x_phi_t_a) / self.dim_context
            )
            g_7_x_phi_t_a = np.einsum(
                "ijk,ij->ik", g_7_x_phi_t_a, time_structure_context
            )
            g_7_x_phi_t_a = g_7_x_phi_t_a @ self.action_context

        # Take the sum of each vector or matrices to consturct h(x, t, a)
        g_x_phi_t_a_ = (
            g_1_x[:, np.newaxis]
            + g_2_phi_t[:, np.newaxis]
            + g_6_phi_t_a
            + g_7_x_phi_t_a
        )

        # q(x, t, a) = \lambda * g(x, \phi(t), a) + (1 - \lambda) * h(x, t, a)
        expected_reward_ = (
            self.lambda_ratio * g_x_phi_t_a_ + (1 - self.lambda_ratio) * h_x_t_a_
        )

        return g_x_phi_t_a_, h_x_t_a_, expected_reward_

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        evaluation_mode=False,
        time_at_evaluation=0,
        time_at_evaluation_vec=None,
        flag_time_at_evaluation=False,
        random_state_for_sampling=None,
    ) -> BanditFeedback:
        check_scalar(n_rounds, "n_rounds", int, min_val=1)

        random_for_sample_ = check_random_state(
            random_state_for_sampling + self.random_state
        )

        # Observe time
        if evaluation_mode == False:
            # Sample time data with size n from the uniform distribution ranging from t_oldest to t_now
            times = random_for_sample_.uniform(
                self.t_oldest, self.t_now, size=n_rounds
            ).astype(int)
        elif flag_time_at_evaluation == False:
            times = time_at_evaluation_vec
        else:
            # All time are time_at_evaluation
            times = np.full(n_rounds, time_at_evaluation)

        # Observe context
        # Stationary context
        if not self.sample_non_stationary_context:
            contexts = random_for_sample_.normal(size=(n_rounds, self.dim_context))
        # Non-stationary context
        else:
            # normalize the time vector
            normalized_time = normalize_time(times, self.t_oldest, self.t_future)

            time_structure_func_for_context_vec = np.vectorize(
                self.time_structure_func_for_context
            )

            time_structures_for_context = time_structure_func_for_context_vec(times)

            time_structure_context_for_context = np.zeros(
                shape=(n_rounds, self.num_time_structure_for_context)
            )

            row_indices = np.arange(n_rounds)
            column_indices = time_structures_for_context

            time_structure_context_for_context[row_indices, column_indices] = 1

            mu_1 = time_structure_context_for_context @ self.gamma

            Sigma_1 = 1

            mu_2 = self.delta[0] * normalized_time

            Sigma_2 = 1

            # Augment the mean vector to the matrix
            mu_1_mat = mu_1[:, np.newaxis]
            mu_1_mat = mu_1_mat * np.ones((1, self.dim_context))

            mu_2_mat = mu_2[:, np.newaxis]
            mu_2_mat = mu_2_mat * np.ones((1, self.dim_context))

            # Sample each of the elements to construct the context
            contexts_1 = random_for_sample_.normal(
                size=(n_rounds, self.dim_context), loc=mu_1_mat, scale=Sigma_1
            )
            contexts_2 = random_for_sample_.normal(
                size=(n_rounds, self.dim_context), loc=mu_2_mat, scale=Sigma_2
            )
            # Synthetize the context
            contexts = (
                self.alpha_ratio * contexts_1 + (1 - self.alpha_ratio) * contexts_2
            )

        g_x_phi_t_a_, h_x_t_a_, expected_reward_ = self.synthesize_expected_reward(
            contexts, times
        )

        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            # correct expected_reward_, as we use truncated normal distribution here
            mean = expected_reward_
            a = (self.reward_min - mean) / self.reward_std
            b = (self.reward_max - mean) / self.reward_std
            expected_reward_ = truncnorm.stats(
                a=a, b=b, loc=mean, scale=self.reward_std, moments="m"
            )

        # calculate the action choice probabilities of the behavior policy
        if self.behavior_policy_function is None:
            pi_b_logits = expected_reward_
        else:
            pi_b_logits = self.behavior_policy_function(
                context=contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        # create some deficient actions based on the value of `n_deficient_actions`
        if self.n_deficient_actions > 0:
            pi_b = np.zeros_like(pi_b_logits)
            n_supported_actions = self.n_actions - self.n_deficient_actions
            supported_actions = np.argsort(
                random_for_sample_.gumbel(size=(n_rounds, self.n_actions)), axis=1
            )[:, ::-1][:, :n_supported_actions]
            supported_actions_idx = (
                np.tile(np.arange(n_rounds), (n_supported_actions, 1)).T,
                supported_actions,
            )
            pi_b[supported_actions_idx] = softmax(
                self.beta * pi_b_logits[supported_actions_idx]
            )
        else:
            pi_b = softmax(self.beta * pi_b_logits)
        # sample actions for each round based on the behavior policy
        actions = sample_action_fast(pi_b, random_state=self.random_state)

        # sample rewards based on the context and action
        rewards = self.sample_reward_given_expected_reward(expected_reward_, actions)

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=contexts,
            time=times,
            action_context=self.action_context,
            action=actions,
            position=None,
            reward=rewards,
            expected_reward=expected_reward_,
            g_x_phi_t_a=g_x_phi_t_a_,
            h_x_t_a=h_x_t_a_,
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(n_rounds), actions],
        )

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, action_dist: np.ndarray
    ) -> float:
        return np.average(expected_reward, weights=action_dist, axis=1).mean()
