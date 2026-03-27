# Copyright (c) 2026 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
from dataclasses import dataclass
from itertools import permutations, product
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from obp.dataset import (
    linear_behavior_policy,
    linear_reward_function,
    logistic_reward_function,
)
from obp.dataset.base import BaseBanditDataset
from obp.types import BanditFeedback

# from obp.utils import check_array
from obp.utils import sigmoid, softmax
from scipy.special import logit, perm
from scipy.stats import rankdata, truncnorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state, check_scalar
from tqdm import tqdm


def check_array(
    array: np.ndarray,
    name: str,
    expected_dim: int = 1,
) -> ValueError:
    """Input validation on an array.

    Parameters
    -------------
    array: object
        Input object to check.

    name: str
        Name of the input array.

    expected_dim: int, default=1
        Expected dimension of the input array.

    """
    if not isinstance(array, np.ndarray):
        raise ValueError(
            f"`{name}` must be {expected_dim}D array, but got {type(array)}"
        )
    if array.ndim != expected_dim:
        raise ValueError(
            f"`{name}` must be {expected_dim}D array, but got {array.ndim}D array"
        )


def gen_eps_greedy(
    expected_reward: np.ndarray,
    is_optimal: bool = True,
    k: int = 1,
    eps: float = 0.0,
) -> np.ndarray:
    "Generate an evaluation policy via the epsilon-greedy rule."
    if is_optimal:
        rank = rankdata(-expected_reward, axis=1)
    else:
        rank = rankdata(expected_reward, axis=1)
    is_topk = rank <= k
    action_dist = ((1.0 - eps) / k) * is_topk
    action_dist += eps / expected_reward.shape[1]
    action_dist /= action_dist.sum(1)[:, np.newaxis]

    # return action_dist[:, :, np.newaxis]
    return action_dist


@dataclass
class RealSlateBanditDataset(BaseBanditDataset):
    n_unique_action: int
    len_list: int
    dim_context: int = 1
    reward_type: str = "binary"
    reward_structure: str = "cascade_additive"
    decay_function: str = "exponential"
    click_model: Optional[str] = None
    eta: float = 1.0
    base_reward_function: Optional[
        Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ]
    ] = None
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    is_factorizable: bool = False
    random_state: int = 12345
    dataset_name: str = "synthetic_slate_bandit_dataset"
    reward_type_conversion: str = "continuous"
    reward_structure_conversion: str = "independent"
    base_reward_function_conversion: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    deterministic_user_threshold: float = 0.0
    effect_from_ranking: float = 0.0
    n_components: int = 10
    threshold: float = 2.0

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_unique_action, "n_unique_action", int, min_val=2)
        if self.is_factorizable:
            max_len_list = None
        else:
            max_len_list = self.n_unique_action
        check_scalar(self.len_list, "len_list", int, min_val=2, max_val=max_len_list)

        check_scalar(self.dim_context, "dim_context", int, min_val=1)
        self.random_ = check_random_state(self.random_state)
        if self.reward_type not in [
            "binary",
            "continuous",
        ]:
            raise ValueError(
                f"`reward_type` must be either 'binary' or 'continuous', but {self.reward_type} is given."
            )
        if self.reward_structure not in [
            "cascade_additive",
            "cascade_decay",
            "independent",
            "standard_additive",
            "standard_decay",
        ]:
            raise ValueError(
                f"`reward_structure` must be one of 'cascade_additive', 'cascade_decay', 'independent', 'standard_additive', or 'standard_decay', but {self.reward_structure} is given."
            )
        if self.decay_function not in ["exponential", "inverse"]:
            raise ValueError(
                f"`decay_function` must be either 'exponential' or 'inverse', but {self.decay_function} is given"
            )
        if self.click_model not in ["cascade", "pbm", None]:
            raise ValueError(
                f"`click_model` must be one of 'cascade', 'pbm', or None, but {self.click_model} is given."
            )
        # set exam_weight (slot-level examination probability).
        # When click_model is 'pbm', exam_weight is :math:`(1 / k)^{\\eta}`, where :math:`k` is the position.
        if self.click_model == "pbm":
            check_scalar(self.eta, name="eta", target_type=float, min_val=0.0)
            self.exam_weight = (1.0 / np.arange(1, self.len_list + 1)) ** self.eta
            self.attractiveness = np.ones(self.len_list, dtype=float)
        elif self.click_model == "cascade":
            check_scalar(self.eta, name="eta", target_type=float, min_val=0.0)
            self.attractiveness = (1.0 / np.arange(1, self.len_list + 1)) ** self.eta
            self.exam_weight = np.ones(self.len_list, dtype=float)
        else:
            self.attractiveness = np.ones(self.len_list, dtype=float)
            self.exam_weight = np.ones(self.len_list, dtype=float)
        if self.click_model is not None and self.reward_type == "continuous":
            raise ValueError(
                "continuous rewards cannot be used when `click_model` is given"
            )
        if self.base_reward_function is not None:
            self.reward_function = action_interaction_reward_function
        if self.reward_structure in ["cascade_additive", "standard_additive"]:
            # generate additive action interaction weight matrix of (n_unique_action, n_unique_action)
            self.action_interaction_weight_matrix = generate_symmetric_matrix(
                n_unique_action=self.n_unique_action, random_state=self.random_state
            )
        else:
            # set decay function
            if self.decay_function == "exponential":
                self.decay_function = exponential_decay_function
            else:  # "inverse"
                self.decay_function = inverse_decay_function
            # generate decay action interaction weight matrix of (len_list, len_list)
            if self.reward_structure == "standard_decay":
                self.action_interaction_weight_matrix = (
                    self.obtain_standard_decay_action_interaction_weight_matrix(
                        self.len_list
                    )
                )
            elif self.reward_structure == "cascade_decay":
                self.action_interaction_weight_matrix = (
                    self.obtain_cascade_decay_action_interaction_weight_matrix(
                        self.len_list
                    )
                )
            else:
                self.action_interaction_weight_matrix = np.zeros(
                    (self.len_list, self.len_list)
                )
        if self.behavior_policy_function is None:
            self.uniform_behavior_policy = (
                np.ones(self.n_unique_action) / self.n_unique_action
            )
        if self.reward_type == "continuous":
            self.reward_min = 0
            self.reward_max = 1e10
            self.reward_std = 3.0
        # one-hot encoding characterizing each action
        self.action_context = np.eye(self.n_unique_action, dtype=int)

        # conversion
        if self.reward_structure_conversion in [
            "cascade_additive",
            "standard_additive",
        ]:
            # generate additive action interaction weight matrix of (n_unique_action, n_unique_action)
            self.action_interaction_weight_matrix_conversion = (
                generate_symmetric_matrix_conversion(
                    n_unique_action=self.n_unique_action,
                    random_state=self.random_state + 555,
                )
            )
        else:
            # set decay function
            if self.decay_function == "exponential":
                self.decay_function_conversion = exponential_decay_function
            else:  # "inverse"
                self.decay_function_conversion = inverse_decay_function
            # generate decay action interaction weight matrix of (len_list, len_list)
            if self.reward_structure_conversion == "standard_decay":
                self.action_interaction_weight_matrix_conversion = (
                    self.obtain_standard_decay_action_interaction_weight_matrix(
                        self.len_list
                    )
                )
            elif self.reward_structure_conversion == "cascade_decay":
                self.action_interaction_weight_matrix_conversion = (
                    self.obtain_cascade_decay_action_interaction_weight_matrix(
                        self.len_list
                    )
                )
            else:
                self.action_interaction_weight_matrix_conversion = np.zeros(
                    (self.len_list, self.len_list)
                )

        if self.reward_type_conversion == "continuous":
            self.reward_min_coversion = 0
            self.reward_max_coversion = 1e10
            self.reward_std_coversion = 1.0

        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.sc = StandardScaler()
        (
            self.fixed_context,
            self.fixed_expected_reward_conversion,
            self.fixed_expected_reward_click,
        ) = self.pre_process()

    def pre_process(
        self,
    ):
        """Preprocess raw dataset."""
        df_small_matrix = pd.read_csv(
            "/Users/kouichi/Desktop/sony/real/data/small_matrix.csv"
        )
        df_user_feature = pd.read_csv(
            "/Users/kouichi/Desktop/sony/real/data/user_features.csv"
        )

        # small_matrix
        small_user_id = df_small_matrix["user_id"]
        user_idx = list(set(small_user_id))
        user_idx = sorted(user_idx)

        small_video_id = df_small_matrix["video_id"]
        video_idx = list(set(small_video_id))
        video_idx = sorted(video_idx)

        small_watch_ratio = df_small_matrix["watch_ratio"]

        small_matrix = np.zeros(7176 * 10728).reshape(7176, 10728)

        small_matrix[small_user_id, small_video_id] = small_watch_ratio

        use_video_id = sorted(
            self.random_.choice(video_idx, self.n_unique_action, replace=False)
        )
        small_matrix = small_matrix[user_idx, :]
        small_matrix = small_matrix[:, use_video_id]

        # click
        base_click = np.ones(small_matrix.shape)
        base_click[small_matrix < self.threshold] = 0.0
        eta = self.random_.uniform(low=0.0, high=0.5, size=base_click.shape)

        fixed_expected_reward_click = base_click * (1 - eta)
        fixed_expected_reward_click += (1 - base_click) * eta

        # contexts
        delete_column = df_user_feature.columns.values[df_user_feature.dtypes == object]
        df_user_feature = df_user_feature.drop(delete_column, axis=1)
        df_user_feature = df_user_feature.dropna(axis=1)
        contexts = np.array(df_user_feature.iloc[user_idx, 1:])
        contexts = self.sc.fit_transform(self.pca.fit_transform(contexts))
        # print(small_matrix.shape)
        return contexts, small_matrix, fixed_expected_reward_click

    def obtain_standard_decay_action_interaction_weight_matrix(
        self,
        len_list,
    ) -> np.ndarray:
        """Obtain an action interaction weight matrix for standard decay reward structure (symmetric matrix)"""
        action_interaction_weight_matrix = np.identity(len_list)
        for pos_ in np.arange(len_list):
            action_interaction_weight_matrix[:, pos_] = -self.decay_function(
                np.abs(np.arange(len_list) - pos_)
            )
            action_interaction_weight_matrix[pos_, pos_] = 0
        return action_interaction_weight_matrix

    def obtain_cascade_decay_action_interaction_weight_matrix(
        self,
        len_list,
    ) -> np.ndarray:
        """Obtain an action interaction weight matrix for cascade decay reward structure (upper triangular matrix)"""
        action_interaction_weight_matrix = np.identity(len_list)
        for pos_ in np.arange(len_list):
            action_interaction_weight_matrix[:, pos_] = -self.decay_function(
                np.abs(np.arange(len_list) - pos_)
            )
            for pos_2 in np.arange(len_list):
                if pos_ <= pos_2:
                    action_interaction_weight_matrix[pos_2, pos_] = 0
        return action_interaction_weight_matrix

    def _calc_pscore_given_policy_logit(
        self,
        all_slate_actions: np.ndarray,
        policy_logit_i_: np.ndarray,
        is_deterministic: bool = False,
    ) -> np.ndarray:
        """Calculate the propensity score of all possible slate actions given a particular policy_logit.

        Parameters
        ------------
        all_slate_actions: array-like, (n_action, len_list)
            All possible slate actions.

        policy_logit_i_: array-like, (n_unique_action, )
            Logit values given context (:math:`x`), which defines the distribution over actions of the policy.

        Returns
        ------------
        pscores: array-like, (n_action, )
            Propensity scores of all slate actions.

        """
        n_actions = len(all_slate_actions)
        unique_action_set_2d = np.tile(np.arange(self.n_unique_action), (n_actions, 1))
        pscores = np.ones(n_actions)
        for pos_ in np.arange(self.len_list):
            action_index = np.where(
                unique_action_set_2d == all_slate_actions[:, pos_][:, np.newaxis]
            )[1]
            if is_deterministic == True:
                pscores *= gen_eps_greedy(
                    policy_logit_i_[unique_action_set_2d], eps=0.0
                )[np.arange(n_actions), action_index]
            else:
                pscores *= softmax(policy_logit_i_[unique_action_set_2d])[
                    np.arange(n_actions), action_index
                ]
            # delete actions
            if pos_ + 1 != self.len_list:
                mask = np.ones((n_actions, self.n_unique_action - pos_))
                mask[np.arange(n_actions), action_index] = 0
                unique_action_set_2d = unique_action_set_2d[mask.astype(bool)].reshape(
                    (-1, self.n_unique_action - pos_ - 1)
                )

        return pscores

    def _calc_pscore_given_policy_softmax(
        self, all_slate_actions: np.ndarray, policy_softmax_i_: np.ndarray
    ) -> np.ndarray:
        """Calculate the propensity score of all possible slate actions given a particular policy_softmax.

        Parameters
        ------------
        all_slate_actions: array-like, (n_action, len_list)
            All possible slate actions.

        policy_softmax_i_: array-like, (n_unique_action, )
            Policy softmax values given context (:math:`x`).

        Returns
        ------------
        pscores: array-like, (n_action, )
            Propensity scores of all slate actions.

        """
        n_actions = len(all_slate_actions)
        unique_action_set_2d = np.tile(np.arange(self.n_unique_action), (n_actions, 1))
        pscores = np.ones(n_actions)
        for pos_ in np.arange(self.len_list):
            action_index = np.where(
                unique_action_set_2d == all_slate_actions[:, pos_][:, np.newaxis]
            )[1]
            score_ = policy_softmax_i_[unique_action_set_2d]
            pscores *= np.divide(score_, score_.sum(axis=1, keepdims=True))[
                np.arange(n_actions), action_index
            ]
            # delete actions
            if pos_ + 1 != self.len_list:
                mask = np.ones((n_actions, self.n_unique_action - pos_))
                mask[np.arange(n_actions), action_index] = 0
                unique_action_set_2d = unique_action_set_2d[mask.astype(bool)].reshape(
                    (-1, self.n_unique_action - pos_ - 1)
                )

        return pscores

    def obtain_pscore_given_evaluation_policy_logit(
        self,
        action: np.ndarray,
        evaluation_policy_logit_: np.ndarray,
        return_pscore_item_position: bool = True,
        clip_logit_value: Optional[float] = None,
    ):
        """Calculate the propensity score given particular logit values to define the evaluation policy.

        Parameters
        ------------
        action: array-like, (n_rounds * len_list, )
            Action chosen by the behavior policy.

        evaluation_policy_logit_: array-like, (n_rounds, n_unique_action)
            Logit values to define the evaluation policy.

        return_pscore_item_position: bool, default=True
            Whether to compute `pscore_item_position` and include it in the logged data.
            When `n_actions` and `len_list` are large, `return_pscore_item_position`=True can lead to a long computation time.

        clip_logit_value: Optional[float], default=None
            A float parameter used to clip logit values (<= `700.`).
            When None, clipping is not applied to softmax values when obtaining `pscore_item_position`.
            When a float value is given, logit values are clipped when calculating softmax values.
            When `n_actions` and `len_list` are large, `clip_logit_value`=None can lead to a long computation time.

        """
        check_array(array=action, name="action", expected_dim=1)
        check_array(
            array=evaluation_policy_logit_,
            name="evaluation_policy_logit_",
            expected_dim=2,
        )
        if (
            len(action) / self.len_list != len(evaluation_policy_logit_)
            or evaluation_policy_logit_.shape[1] != self.n_unique_action
        ):
            raise ValueError(
                "the shape of `action` and `evaluation_policy_logit_` must be (n_rounds * len_list, )"
                "and (n_rounds, n_unique_action) respectively"
            )

        n_rounds = action.reshape((-1, self.len_list)).shape[0]
        pscore_cascade = np.zeros(n_rounds * self.len_list)
        pscore = np.zeros(n_rounds * self.len_list)
        if return_pscore_item_position:
            pscore_item_position = np.zeros(n_rounds * self.len_list)
            if not self.is_factorizable:
                enumerated_slate_actions = [
                    _
                    for _ in permutations(
                        np.arange(self.n_unique_action), self.len_list
                    )
                ]
                enumerated_slate_actions = np.array(enumerated_slate_actions)
        else:
            pscore_item_position = None
        if return_pscore_item_position and clip_logit_value is not None:
            check_scalar(
                clip_logit_value,
                name="clip_logit_value",
                target_type=(float),
                max_val=700.0,
            )
            evaluation_policy_softmax_ = np.exp(
                np.minimum(evaluation_policy_logit_, clip_logit_value)
            )
        # for i in tqdm(
        #     np.arange(n_rounds),
        #     desc="[obtain_pscore_given_evaluation_policy_logit]",
        #     total=n_rounds,
        # ):
        for i in np.arange(n_rounds):
            unique_action_set = np.arange(self.n_unique_action)
            score_ = softmax(evaluation_policy_logit_[i : i + 1])[0]
            pscore_i = 1.0
            for pos_ in np.arange(self.len_list):
                action_ = action[i * self.len_list + pos_]
                action_index_ = np.where(unique_action_set == action_)[0][0]
                # calculate joint pscore
                pscore_i *= score_[action_index_]
                pscore_cascade[i * self.len_list + pos_] = pscore_i
                # update the pscore given the remaining items for nonfactorizable policy
                if not self.is_factorizable and pos_ != self.len_list - 1:
                    unique_action_set = np.delete(
                        unique_action_set, unique_action_set == action_
                    )
                    score_ = softmax(
                        evaluation_policy_logit_[i : i + 1, unique_action_set]
                    )[0]
                # calculate pscore_item_position
                if return_pscore_item_position:
                    if pos_ == 0:
                        pscore_item_pos_i_l = pscore_i
                    elif self.is_factorizable:
                        pscore_item_pos_i_l = score_[action_index_]
                    else:
                        if isinstance(clip_logit_value, float):
                            pscores = self._calc_pscore_given_policy_softmax(
                                all_slate_actions=enumerated_slate_actions,
                                policy_softmax_i_=evaluation_policy_softmax_[i],
                            )
                        else:
                            pscores = self._calc_pscore_given_policy_logit(
                                all_slate_actions=enumerated_slate_actions,
                                policy_logit_i_=evaluation_policy_logit_[i],
                            )
                        pscore_item_pos_i_l = pscores[
                            enumerated_slate_actions[:, pos_] == action_
                        ].sum()
                    pscore_item_position[i * self.len_list + pos_] = pscore_item_pos_i_l
            # impute joint pscore
            start_idx = i * self.len_list
            end_idx = start_idx + self.len_list
            pscore[start_idx:end_idx] = pscore_i

        return pscore, pscore_item_position, pscore_cascade

    def sample_action_and_obtain_pscore(
        self,
        behavior_policy_logit_: np.ndarray,
        n_rounds: int,
        context: np.ndarray,
        return_pscore_item_position: bool = True,
        clip_logit_value: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Sample action and obtain the three variants of the propensity scores.

        Parameters
        ------------
        behavior_policy_logit_: array-like, shape (n_rounds, n_actions)
            Logit values given context (:math:`x`).

        n_rounds: int
            Data size of synthetic logged data.

        return_pscore_item_position: bool, default=True
            Whether to compute `pscore_item_position` and include it in the logged data.
            When `n_actions` and `len_list` are large, `return_pscore_item_position`=True can lead to a long computation time.

        clip_logit_value: Optional[float], default=None
            A float parameter used to clip logit values (<= `700.`).
            When None, clipping is not applied to softmax values when obtaining `pscore_item_position`.
            When a float value is given, logit values are clipped when calculating softmax values.
            When `n_actions` and `len_list` are large, `clip_logit_value`=None can lead to a long computation time.

        Returns
        ----------
        action: array-like, shape (n_rounds * len_list)
            Actions sampled by the behavior policy.
            Actions sampled within slate `i` is stored in `action[`i` * `len_list`: (`i + 1`) * `len_list`]`.

        pscore: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the slate actions given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,1}, a_{i,2}, \\ldots, a_{i,L} | x_{i} )`.

        pscore_item_position: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the action of the :math:`l`-th slot given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,l} | x_{i} )`.

        pscore_cascade: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the actions of the top :math:`l` slots given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,1}, a_{i,2}, \\ldots, a_{i,l} | x_{i} )`.

        """
        action = np.zeros(n_rounds * self.len_list, dtype=int)
        pscore_cascade = np.zeros(n_rounds * self.len_list)
        pscore = np.zeros(n_rounds * self.len_list)
        p_click_pi_0 = np.zeros((n_rounds, self.n_unique_action))  # pc(x,a,\pi_0)
        p_click_factual_pi_0 = np.zeros(n_rounds * self.len_list)  # pc(x,a,\pi_0)

        if return_pscore_item_position:
            pscore_item_position = np.zeros(n_rounds * self.len_list)
            if not self.is_factorizable and self.behavior_policy_function is not None:
                enumerated_slate_actions = [
                    _
                    for _ in permutations(
                        np.arange(self.n_unique_action), self.len_list
                    )
                ]
                enumerated_slate_actions = np.array(enumerated_slate_actions)
        else:
            pscore_item_position = None
        if return_pscore_item_position and clip_logit_value is not None:
            check_scalar(
                clip_logit_value,
                name="clip_logit_value",
                target_type=(float),
                max_val=700.0,
            )
            behavior_policy_softmax_ = np.exp(
                np.minimum(behavior_policy_logit_, clip_logit_value)
            )
        # for i in tqdm(
        #     np.arange(n_rounds),
        #     desc="[sample_action_and_obtain_pscore]",
        #     total=n_rounds,
        # ):
        for i in np.arange(n_rounds):
            is_deterministic = False
            unique_action_set = np.arange(self.n_unique_action)

            if context[i][0] <= self.deterministic_user_threshold:
                is_deterministic = True
                score_ = gen_eps_greedy(
                    behavior_policy_logit_[i : i + 1, unique_action_set], eps=0.0
                ).reshape(-1)
            else:
                score_ = softmax(behavior_policy_logit_[i : i + 1, unique_action_set])[
                    0
                ]

            pscore_i = 1.0
            pscores = 0
            action_for_i = np.zeros(self.len_list)
            for pos_ in np.arange(self.len_list):
                sampled_action = self.random_.choice(
                    unique_action_set, p=score_, replace=False
                )
                action[i * self.len_list + pos_] = sampled_action
                sampled_action_index = np.where(unique_action_set == sampled_action)[0][
                    0
                ]
                action_for_i[pos_] = sampled_action
                # calculate joint pscore
                pscore_i *= score_[sampled_action_index]
                pscore_cascade[i * self.len_list + pos_] = pscore_i
                # update the pscore given the remaining items for nonfactorizable behavior policy
                if not self.is_factorizable and pos_ != self.len_list - 1:
                    unique_action_set = np.delete(
                        unique_action_set, unique_action_set == sampled_action
                    )
                    if context[i][0] <= self.deterministic_user_threshold:
                        score_ = gen_eps_greedy(
                            behavior_policy_logit_[i : i + 1, unique_action_set],
                            eps=0.0,
                        ).reshape(-1)
                    else:
                        score_ = softmax(
                            behavior_policy_logit_[i : i + 1, unique_action_set]
                        )[0]
                # calculate pscore_item_position
                if return_pscore_item_position:
                    if self.behavior_policy_function is None:  # uniform random
                        pscore_item_pos_i_l = 1 / self.n_unique_action
                    elif self.is_factorizable:
                        pscore_item_pos_i_l = score_[sampled_action_index]
                    elif pos_ == 0:
                        pscore_item_pos_i_l = pscore_i
                    else:
                        if isinstance(clip_logit_value, float):
                            pscores = self._calc_pscore_given_policy_softmax(
                                all_slate_actions=enumerated_slate_actions,
                                policy_softmax_i_=behavior_policy_softmax_[i],
                            )
                        else:
                            if pos_ == 1:
                                pscores = self._calc_pscore_given_policy_logit(
                                    all_slate_actions=enumerated_slate_actions,
                                    policy_logit_i_=behavior_policy_logit_[i],
                                    is_deterministic=is_deterministic,
                                )
                                # print(f"{i}",behavior_policy_logit_[i,:5])
                                # print(f"{i, pos_}",pscores[:5])
                            else:
                                assert pscores is not None, (
                                    "pscores should have been calculated at pos_ == 0"
                                )
                        # print(f"{i, pos_}",pscores[:5])
                        pscore_item_pos_i_l = pscores[
                            enumerated_slate_actions[:, pos_] == sampled_action
                        ].sum()
                    pscore_item_position[i * self.len_list + pos_] = pscore_item_pos_i_l
            expected_reward_all_click = self.reward_function(
                context=context[i].reshape(1, -1),
                action_context=self.action_context,
                action=enumerated_slate_actions.flatten(),
                action_interaction_weight_matrix=self.action_interaction_weight_matrix,
                expected_reward=self.expected_reward_click,
                reward_type=self.reward_type,
                reward_structure=self.reward_structure,
                len_list=self.len_list,
                is_enumerated=True,
                random_state=self.random_state,
            )
            # p_click_pi_0 for all unique action
            for a in np.arange(self.n_unique_action):
                idx = np.where(enumerated_slate_actions == a)
                p_A = pscores[idx[0]]
                q_x_c = expected_reward_all_click[idx]
                p_click_pi_0[i, a] = (p_A * q_x_c).sum()

            p_click_factual_pi_0[
                i * self.len_list : i * self.len_list + self.len_list
            ] = p_click_pi_0[i, action_for_i.astype(int)]

            # impute joint pscore
            start_idx = i * self.len_list
            end_idx = start_idx + self.len_list
            pscore[start_idx:end_idx] = pscore_i

            # print(pscore)

        return (
            action,
            pscore_cascade,
            pscore,
            pscore_item_position,
            p_click_factual_pi_0,
        )

    def sample_contextfree_expected_reward(
        self, random_state: Optional[int] = None
    ) -> np.ndarray:
        """Define context independent expected rewards for each action and slot.

        Parameters
        -----------
        random_state: int, default=None
            Controls the random seed in sampling dataset.

        """
        random_ = check_random_state(random_state)
        return random_.uniform(size=(self.n_unique_action, self.len_list))

    def sample_reward_given_expected_reward(
        self,
        expected_reward_factual: np.ndarray,
        expected_reward_factual_click: np.ndarray,
        expected_reward_factual_conversion: np.ndarray,
    ) -> np.ndarray:
        """Sample reward variables given actions observed at each slot.

        Parameters
        ------------
        expected_reward_factual: array-like, shape (n_rounds, len_list)
            Expected rewards given observed actions and contexts.

        Returns
        ----------
        reward: array-like, shape (n_rounds, len_list)
            Sampled rewards.

        """
        # click
        if self.reward_type == "binary":
            sampled_reward_list = list()
            for pos_ in np.arange(self.len_list):
                expected_reward_factual_at_position = expected_reward_factual_click[
                    :, pos_
                ]

                sampled_rewards_at_position = self.random_.binomial(
                    n=1, p=expected_reward_factual_at_position
                )
                sampled_reward_list.append(sampled_rewards_at_position)
            reward_click = np.array(sampled_reward_list).T
        elif self.reward_type == "continuous":
            reward_click = np.zeros(expected_reward_factual_click.shape)
            for pos_ in np.arange(self.len_list):
                mean = expected_reward_factual_click[:, pos_]

                reward_click[:, pos_] = self.random_.normal(
                    loc=mean,
                    scale=self.reward_std,
                )
        else:
            raise NotImplementedError

        # conversion
        if self.reward_type_conversion == "binary":
            sampled_reward_list = list()
            for pos_ in np.arange(self.len_list):
                expected_reward_factual_at_position = (
                    expected_reward_factual_conversion[:, pos_]
                )

                sampled_rewards_at_position = self.random_.binomial(
                    n=1, p=expected_reward_factual_at_position
                )
                sampled_reward_list.append(sampled_rewards_at_position)
            reward_conversion = np.array(sampled_reward_list).T
        elif self.reward_type_conversion == "continuous":
            reward_conversion = np.zeros(expected_reward_factual_click.shape)
            for pos_ in np.arange(self.len_list):
                mean = expected_reward_factual_conversion[:, pos_]

                reward_conversion[:, pos_] = self.random_.normal(
                    loc=mean,
                    scale=self.reward_std_coversion,
                )
        else:
            raise NotImplementedError

        reward = reward_click * reward_conversion
        return reward
        # expected_reward_factual *= self.exam_weight
        # if self.reward_type == "binary":
        #     sampled_reward_list = list()
        #     discount_factors = np.ones(expected_reward_factual.shape[0])
        #     sampled_rewards_at_position = np.zeros(expected_reward_factual.shape[0])
        #     for pos_ in np.arange(self.len_list):
        #         discount_factors *= sampled_rewards_at_position * self.attractiveness[
        #             pos_
        #         ] + (1 - sampled_rewards_at_position)
        #         expected_reward_factual_at_position = (
        #             discount_factors * expected_reward_factual[:, pos_]
        #         )
        #         sampled_rewards_at_position = self.random_.binomial(
        #             n=1, p=expected_reward_factual_at_position
        #         )
        #         sampled_reward_list.append(sampled_rewards_at_position)
        #     reward = np.array(sampled_reward_list).T

        # elif self.reward_type == "continuous":
        #     reward = np.zeros(expected_reward_factual.shape)
        #     for pos_ in np.arange(self.len_list):
        #         mean = expected_reward_factual[:, pos_]
        #         # print(mean)
        #         # a = (self.reward_min - mean) / self.reward_std
        #         # b = (self.reward_max - mean) / self.reward_std
        #         # reward[:, pos_] = truncnorm.rvs(
        #         #     a=a,
        #         #     b=b,
        #         #     loc=mean,
        #         #     scale=self.reward_std,
        #         #     random_state=self.random_state,
        #         # )
        #         reward[:, pos_] = self.random_.normal(
        #             loc=mean,
        #             scale=self.reward_std,
        #         )
        # else:
        #     raise NotImplementedError
        # # return: array-like, shape (n_rounds, len_list)
        # return reward

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        return_pscore_item_position: bool = True,
        clip_logit_value: Optional[float] = None,
    ) -> BanditFeedback:
        """Obtain batch logged bandit data.

        Parameters
        ----------
        n_rounds: int
            Data size of the synthetic logged bandit data.

        return_pscore_item_position: bool, default=True
            Whether to compute `pscore_item_position` and include it in the logged data.
            When `n_unique_action` and `len_list` are large, this should be set to False due to computation time.

        clip_logit_value: Optional[float], default=None
            A float parameter to clip logit values.
            When None, we calculate softmax values without clipping to obtain `pscore_item_position`.
            When a float value is given, we clip logit values to calculate softmax values to obtain `pscore_item_position`.
            When `n_actions` and `len_list` are large, `clip_logit_value`=None can lead to a long computation time.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            Synthesized slate logged bandit dataset.

        """
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        user_idx = self.random_.choice(self.fixed_context.shape[0], size=n_rounds)
        context = self.fixed_context[user_idx]

        self.expected_reward_click = self.fixed_expected_reward_click[user_idx]
        self.expected_reward_conversion = self.fixed_expected_reward_conversion[
            user_idx
        ]
        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is None:
            behavior_policy_logit_ = np.tile(
                self.uniform_behavior_policy, (n_rounds, 1)
            )
        else:
            behavior_policy_logit_ = self.behavior_policy_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        # check the shape of behavior_policy_logit_
        if not (
            isinstance(behavior_policy_logit_, np.ndarray)
            and behavior_policy_logit_.shape == (n_rounds, self.n_unique_action)
        ):
            raise ValueError("`behavior_policy_logit_` has an invalid shape")
        # sample actions and calculate the three variants of the propensity scores
        (
            action,
            pscore_cascade,
            pscore,
            pscore_item_position,
            p_click_factual_pi_0,
        ) = self.sample_action_and_obtain_pscore(
            behavior_policy_logit_=behavior_policy_logit_,
            n_rounds=n_rounds,
            return_pscore_item_position=return_pscore_item_position,
            clip_logit_value=clip_logit_value,
            context=context,
        )
        # sample expected reward factual
        if self.base_reward_function is None:
            expected_reward = self.sample_contextfree_expected_reward(
                random_state=self.random_state
            )
            expected_reward_tile = np.tile(expected_reward, (n_rounds, 1, 1))
            # action_2d: array-like, shape (n_rounds, len_list)
            action_2d = action.reshape((n_rounds, self.len_list))
            # expected_reward_factual: array-like, shape (n_rounds, len_list)
            expected_reward_factual = np.array(
                [
                    expected_reward_tile[np.arange(n_rounds), action_2d[:, pos_], pos_]
                    for pos_ in np.arange(self.len_list)
                ]
            ).T
        else:
            # expected_reward_fuctual (n_rounds, len_list)
            expected_reward_factual_click = self.reward_function(
                context=context,
                action_context=self.action_context,
                action=action,
                action_interaction_weight_matrix=self.action_interaction_weight_matrix,
                expected_reward=self.expected_reward_click,
                reward_type=self.reward_type,
                reward_structure=self.reward_structure,
                len_list=self.len_list,
                random_state=self.random_state,
            )
            expected_reward_factual_conversion = action_interaction_reward_function_conversion(
                context=context,
                action_context=self.action_context,
                action=action,
                action_interaction_weight_matrix=self.action_interaction_weight_matrix_conversion,
                expected_reward=self.expected_reward_conversion,
                reward_type=self.reward_type_conversion,
                reward_structure=self.reward_structure_conversion,
                len_list=self.len_list,
                random_state=self.random_state + 555,
                effect_from_ranking=self.effect_from_ranking,
            )
            expected_reward_factual = (
                expected_reward_factual_click * expected_reward_factual_conversion
            )
        # check the shape of expected_reward_factual
        if not (
            isinstance(expected_reward_factual, np.ndarray)
            and expected_reward_factual.shape == (n_rounds, self.len_list)
        ):
            raise ValueError("`expected_reward_factual` has an invalid shape")
        # sample reward (n_rounds, len_list)
        reward = self.sample_reward_given_expected_reward(
            expected_reward_factual=expected_reward_factual,
            expected_reward_factual_click=expected_reward_factual_click,
            expected_reward_factual_conversion=expected_reward_factual_conversion,
        )
        return dict(
            n_rounds=n_rounds,
            user_idx=user_idx,
            n_unique_action=self.n_unique_action,
            slate_id=np.repeat(np.arange(n_rounds), self.len_list),
            context=context,
            action_context=self.action_context,
            action=action,
            position=np.tile(np.arange(self.len_list), n_rounds),
            reward=reward.reshape(action.shape[0]),
            reward_click=(reward.reshape(action.shape[0]) >= 0).astype(int),
            expected_reward_factual=expected_reward_factual.reshape(action.shape[0]),
            expected_reward_factual_click=expected_reward_factual_click.reshape(
                action.shape[0]
            ),
            expected_reward_factual_conversion=expected_reward_factual_conversion.reshape(
                action.shape[0]
            ),
            pscore_cascade=pscore_cascade,
            pscore=pscore,
            pscore_item_position=pscore_item_position,
            p_click_factual_pi_0=p_click_factual_pi_0,
        )

    def calc_on_policy_policy_value(
        self, reward: np.ndarray, slate_id: np.ndarray
    ) -> float:
        """Calculate the policy value of given reward and slate_id.

        Parameters
        -----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        slate_id: array-like, shape (<= n_rounds * len_list,)
            Slate index.

        Returns
        ----------
        policy_value: float
            The on-policy policy value estimate of the behavior policy.

        """
        check_array(array=slate_id, name="slate_id", expected_dim=1)
        check_array(array=reward, name="reward", expected_dim=1)
        if reward.shape[0] != slate_id.shape[0]:
            raise ValueError(
                "Expected `reward.shape[0] == slate_id.shape[0]`, but found it False"
            )

        return reward.sum() / np.unique(slate_id).shape[0]

    def calc_ground_truth_policy_value(
        self,
        context: np.ndarray,
        evaluation_policy_logit_: np.ndarray,
    ):
        """Calculate the ground-truth policy value of given evaluation policy logit and contexts.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        evaluation_policy_logit_: array-like, shape (n_rounds, n_unique_action)
            Logit values to define the evaluation policy.

        """
        check_array(array=context, name="context", expected_dim=2)
        check_array(
            array=evaluation_policy_logit_,
            name="evaluation_policy_logit_",
            expected_dim=2,
        )
        if evaluation_policy_logit_.shape[1] != self.n_unique_action:
            raise ValueError(
                "Expected `evaluation_policy_logit_.shape[1] != self.n_unique_action`,"
                "but found it False"
            )
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )
        if evaluation_policy_logit_.shape[0] != context.shape[0]:
            raise ValueError(
                "Expected `evaluation_policy_logit_.shape[0] == context.shape[0]`,"
                "but found it False"
            )

        if self.is_factorizable:
            enumerated_slate_actions = [
                _
                for _ in product(np.arange(self.n_unique_action), repeat=self.len_list)
            ]
        else:
            enumerated_slate_actions = [
                _ for _ in permutations(np.arange(self.n_unique_action), self.len_list)
            ]
        enumerated_slate_actions = np.array(enumerated_slate_actions).astype("int8")
        n_slate_actions = len(enumerated_slate_actions)
        n_rounds = len(evaluation_policy_logit_)

        pscores = []
        n_enumerated_slate_actions = len(enumerated_slate_actions)
        if self.is_factorizable:
            # for action_list in tqdm(
            #     enumerated_slate_actions,
            #     desc="[calc_ground_truth_policy_value (pscore)]",
            #     total=n_enumerated_slate_actions,
            # ):
            for action_list in enumerated_slate_actions:
                pscores.append(
                    softmax(evaluation_policy_logit_)[:, action_list].prod(1)
                )
            pscores = np.array(pscores).T
        else:
            # for i in tqdm(
            #     np.arange(n_rounds),
            #     desc="[calc_ground_truth_policy_value (pscore)]",
            #     total=n_rounds,
            # ):
            for i in np.arange(n_rounds):
                pscores.append(
                    self._calc_pscore_given_policy_logit(
                        all_slate_actions=enumerated_slate_actions,
                        policy_logit_i_=evaluation_policy_logit_[i],
                    )
                )
            pscores = np.array(pscores)

        # calculate expected slate-level reward for each combinatorial set of items (i.e., slate actions)
        if self.base_reward_function is None:
            expected_slot_reward = self.sample_contextfree_expected_reward(
                random_state=self.random_state
            )
            expected_slot_reward_tile = np.tile(
                expected_slot_reward, (n_rounds * n_slate_actions, 1, 1)
            )
            expected_slate_rewards = np.array(
                [
                    expected_slot_reward_tile[
                        np.arange(n_slate_actions) % n_slate_actions,
                        np.array(enumerated_slate_actions)[:, pos_],
                        pos_,
                    ]
                    for pos_ in np.arange(self.len_list)
                ]
            ).T
            policy_value = (pscores * expected_slate_rewards.sum(axis=1)).sum()
        else:
            n_batch = (
                n_rounds * n_enumerated_slate_actions * self.len_list - 1
            ) // 10**7 + 1
            batch_size = (n_rounds - 1) // n_batch + 1
            n_batch = (n_rounds - 1) // batch_size + 1

            policy_value = 0.0
            # for batch_idx in tqdm(
            #     np.arange(n_batch),
            #     desc=f"[calc_ground_truth_policy_value (expected reward), batch_size={batch_size}]",
            #     total=n_batch,
            # ):
            for batch_idx in np.arange(n_batch):
                context_ = context[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]
                pscores_ = pscores[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]

                expected_slate_rewards_ = self.reward_function(
                    context=context_,
                    action_context=self.action_context,
                    action=enumerated_slate_actions.flatten(),
                    action_interaction_weight_matrix=self.action_interaction_weight_matrix,
                    expected_reward=self.expected_reward_click,
                    base_reward_function=self.base_reward_function,
                    reward_type=self.reward_type,
                    reward_structure=self.reward_structure,
                    len_list=self.len_list,
                    is_enumerated=True,
                    random_state=self.random_state,
                )

                # click models based on expected reward
                expected_slate_rewards_ *= self.exam_weight
                if self.reward_type == "binary":
                    discount_factors = np.ones(expected_slate_rewards_.shape[0])
                    previous_slot_expected_reward = np.zeros(
                        expected_slate_rewards_.shape[0]
                    )
                    for pos_ in np.arange(self.len_list):
                        discount_factors *= (
                            previous_slot_expected_reward * self.attractiveness[pos_]
                            + (1 - previous_slot_expected_reward)
                        )
                        expected_slate_rewards_[:, pos_] = (
                            discount_factors * expected_slate_rewards_[:, pos_]
                        )
                        previous_slot_expected_reward = expected_slate_rewards_[:, pos_]

                policy_value += (
                    pscores_.flatten() * expected_slate_rewards_.sum(axis=1)
                ).sum()
            policy_value /= n_rounds

        return policy_value

    def generate_evaluation_policy_pscore(
        self,
        evaluation_policy_type: str,
        context: np.ndarray,
        action: Optional[np.ndarray] = None,
        epsilon: Optional[float] = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate three variants of propensity scores of synthetic evaluation policies.

        Parameters
        -----------
        evaluation_policy_type: str
            Specify the type of evaluation policy to generate, which must be one of 'optimal', 'anti-optimal', or 'random'.
            When 'optimal' is given, we sort actions based on the base expected rewards (outputs of `base_reward_function`) and extract top-L actions (L=`len_list`) for each slate.
            When 'anti-optimal' is given, we sort actions based on the base expected rewards (outputs of `base_reward_function`) and extract bottom-L actions (L=`len_list`) for each slate.
            We calculate the three variants of the propensity scores (pscore, `pscore_item_position`, and pscore_cascade) of the epsilon-greedy policy when either 'optimal' or 'anti-optimal' is given.
            When 'random' is given, we calculate the three variants of the propensity scores of the uniform random policy.

        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds * len_list,), default=None
            Actions sampled by the behavior policy.
            Actions sampled within slate `i` is stored in `action[`i` * `len_list`: (`i + 1`) * `len_list`]`.
            When `evaluation_policy_type`='random', this argument is irrelevant.

        epsilon: float, default=1.
            Exploration hyperparameter that must take value in the range of [0., 1.].
            When `evaluation_policy_type`='random', this argument is irrelevant.

        Returns
        ----------
        pscore: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the slate actions given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,1}, a_{i,2}, \\ldots, a_{i,L} | x_{i} )`.

        pscore_item_position: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the action of the :math:`l`-th slot given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,l} | x_{i} )`.

        pscore_cascade: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the actions of the top :math:`l` slots given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,1}, a_{i,2}, \\ldots, a_{i,l} | x_{i} )`.

        """
        check_array(array=context, name="context", expected_dim=2)
        if evaluation_policy_type not in ["optimal", "anti-optimal", "random"]:
            raise ValueError(
                f"`evaluation_policy_type` must be 'optimal', 'anti-optimal', or 'random', but {evaluation_policy_type} is given"
            )

        # [Caution]: OverflowError raises when integer division result is too large for a float
        if self.is_factorizable:
            random_pscore_cascade = (
                (np.ones((context.shape[0], self.len_list)) / self.n_unique_action)
                .cumprod(axis=1)
                .flatten()
            )
            random_pscore = np.ones(context.shape[0] * self.len_list) / (
                self.n_unique_action**self.len_list
            )
        else:
            random_pscore_cascade = (
                1.0
                / np.tile(
                    np.arange(
                        self.n_unique_action, self.n_unique_action - self.len_list, -1
                    ),
                    (context.shape[0], 1),
                )
                .cumprod(axis=1)
                .flatten()
            )
            random_pscore = np.ones(context.shape[0] * self.len_list) / perm(
                self.n_unique_action, self.len_list
            )
        random_pscore_item_position = (
            np.ones(context.shape[0] * self.len_list) / self.n_unique_action
        )
        if evaluation_policy_type == "random":
            return random_pscore, random_pscore_item_position, random_pscore_cascade

        else:
            # base_expected_reward: array-like, shape (n_rounds, n_unique_action)
            base_expected_reward = self.base_reward_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
            check_array(array=action, name="action", expected_dim=1)
            if action.shape[0] != context.shape[0] * self.len_list:
                raise ValueError(
                    "Expected `action.shape[0] == context.shape[0] * self.len_list`,"
                    "but found it False"
                )
            action_2d = action.reshape((context.shape[0], self.len_list))
            if context.shape[0] != action_2d.shape[0]:
                raise ValueError(
                    "Expected `context.shape[0] == action_2d.shape[0]`, but found it False"
                )

            check_scalar(
                epsilon, name="epsilon", target_type=(float), min_val=0.0, max_val=1.0
            )
            if evaluation_policy_type == "optimal":
                sorted_actions = base_expected_reward.argsort(axis=1)[
                    :, : self.len_list
                ]
            else:
                sorted_actions = base_expected_reward.argsort(axis=1)[
                    :, -self.len_list :
                ]
            (
                pscore,
                pscore_item_position,
                pscore_cascade,
            ) = self._calc_epsilon_greedy_pscore(
                epsilon=epsilon,
                action_2d=action_2d,
                sorted_actions=sorted_actions,
                random_pscore=random_pscore,
                random_pscore_item_position=random_pscore_item_position,
                random_pscore_cascade=random_pscore_cascade,
            )
        return pscore, pscore_item_position, pscore_cascade

    def calc_evaluation_policy_action_dist(
        self,
        action: np.ndarray,
        evaluation_policy_logit_: np.ndarray,
    ):
        """Calculate action distribution at each slot from a given evaluation policy logit.

        Parameters
        ----------
        action: array-like, shape (n_rounds * len_list, )
            Action chosen by behavior policy.

        evaluation_policy_logit_: array-like, shape (n_rounds, n_unique_action)
            Logit values of evaluation policy given context (:math:`x`), i.e., :math:`\\f: \\mathcal{X} \\rightarrow \\mathbb{R}^{\\mathcal{A}}`.

        Returns
        ----------
        evaluation_policy_action_dist: array-like, shape (n_rounds * len_list * n_unique_action, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

        """
        check_array(action, name="action", expected_dim=1)
        check_array(
            evaluation_policy_logit_, name="evaluation_policy_logit_", expected_dim=2
        )
        if evaluation_policy_logit_.shape[1] != self.n_unique_action:
            raise ValueError(
                "Expected `evaluation_policy_logit_.shape[1] == n_unique_action`, but found it False"
            )
        if len(action) != evaluation_policy_logit_.shape[0] * self.len_list:
            raise ValueError(
                "Expected `len(action) == evaluation_policy_logit_.shape[0] * len_list`, but found it False"
            )
        n_rounds = evaluation_policy_logit_.shape[0]

        # (n_rounds * len_list, ) -> (n_rounds, len_list)
        action = action.reshape((n_rounds, self.len_list))
        # (n_rounds, n_unique_action) -> (n_rounds, len_list, n_unique_action)
        evaluation_policy_logit_ = np.array(
            [
                [evaluation_policy_logit_[i] for _ in range(self.len_list)]
                for i in range(n_rounds)
            ]
        )
        # calculate action probabilities for all the counterfactual actions at the position
        # (n_rounds, len_list, n_unique_action)
        evaluation_policy_action_dist = []
        for i in range(n_rounds):
            if not self.is_factorizable:
                for pos_ in range(self.len_list - 1):
                    action_ = action[i][pos_]
                    # mask action choice probability of the previously chosen action
                    # to avoid overflow in softmax function, set -1e4 instead of -np.inf
                    # (make action choice probability 0 for the previously chosen action by softmax)
                    evaluation_policy_logit_[i, pos_ + 1 :, action_] = -1e4
            # (len_list, n_unique_action)
            evaluation_policy_action_dist.append(softmax(evaluation_policy_logit_[i]))
        # (n_rounds, len_list, n_unique_action) -> (n_rounds * len_list * n_unique_action, )
        evaluation_policy_action_dist = np.array(
            evaluation_policy_action_dist
        ).flatten()
        return evaluation_policy_action_dist

    def _calc_epsilon_greedy_pscore(
        self,
        epsilon: float,
        action_2d: np.ndarray,
        sorted_actions: np.ndarray,
        random_pscore: np.ndarray,
        random_pscore_item_position: np.ndarray,
        random_pscore_cascade: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate three variants of the propensity scores of synthetic evaluation policies via epsilon-greedy.

        Parameters
        -----------
        epsilon: float, default=1.
            Exploration hyperparameter in the epsilon-greedy rule.
            Must take value in the range of [0., 1.].

        action_2d: array-like, shape (n_rounds, len_list), default=None
            Actions sampled by the behavior policy.
            Actions sampled within slate `i` is stored in `action[i]`.
            When bandit_feedback is obtained by `obtain_batch_bandit_feedback`, we can obtain action_2d as follows: bandit_feedback["action"].reshape((n_rounds, len_list))
            When `evaluation_policy_type`='random', this argument is unnecessary.

        random_pscore: array-like, shape (n_unique_action * len_list, )
            Probabilities of the uniform random policy choosing the slate actions given context (:math:`x`),
            i.e., :math:`\\pi_{unif} (a_{i,1}, a_{i,2}, \\ldots, a_{i,L} | x_{i} )`.

        random_pscore_item_position: array-like, shape (n_unique_action * len_list, )
            Probabilities of the uniform random policy choosing the action of the :math:`l`-th slot given context (:math:`x`), i.e., :math:`\\pi_{unif}(a_{i,l} | x_{i} )`.

        random_pscore_cascade: array-like, shape (n_unique_action * len_list, )
            Probabilities of the uniform random policy choosing the actions of the top :math:`l` slots given context (:math:`x`), i.e., :math:`\\pi_{unif}(a_{i,1}, a_{i,2}, \\ldots, a_{i,l} | x_{i} )`.

        Returns
        ----------
        pscore: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the slate actions given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,1}, a_{i,2}, \\ldots, a_{i,L} | x_{i} )`.

        pscore_item_position: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the action of the :math:`l`-th slot given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,l} | x_{i} )`.

        pscore_cascade: array-like, shape (n_unique_action * len_list)
            Probabilities of choosing the actions of the top :math:`l` slots given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,1}, a_{i,2}, \\ldots, a_{i,l} | x_{i} )`.

        """
        check_array(array=action_2d, name="action_2d", expected_dim=2)
        if not self.is_factorizable and set(
            [np.unique(x).shape[0] for x in action_2d]
        ) != set([self.len_list]):
            raise ValueError(
                "when `is_factorizable`=False, actions observed within each slate must be unique"
            )
        if self.is_factorizable:
            action_match_flg = (
                np.tile(sorted_actions[:, 0], (action_2d.shape[1], 1)).T == action_2d
            )
        else:
            action_match_flg = sorted_actions == action_2d
        pscore_flg = np.repeat(action_match_flg.all(axis=1), self.len_list)
        pscore_item_position_flg = action_match_flg.flatten()
        pscore_cascade_flg = action_match_flg.cumprod(axis=1).flatten()
        # calculate the three variants of the propensity scores based on the given epsilon value
        pscore = pscore_flg * (1 - epsilon) + epsilon * random_pscore
        pscore_item_position = (
            pscore_item_position_flg * (1 - epsilon)
            + epsilon * random_pscore_item_position
        )
        pscore_cascade = (
            pscore_cascade_flg * (1 - epsilon) + epsilon * random_pscore_cascade
        )
        return pscore, pscore_item_position, pscore_cascade

    def calc_ground_truth_policy_value_epsilon_greedy(
        self,
        context: np.ndarray,
        evaluation_policy_logit_: np.ndarray,
        eps: float,
        user_idx: np.ndarray,
    ):
        """Calculate the ground-truth policy value of given evaluation policy logit and contexts.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        evaluation_policy_logit_: array-like, shape (n_rounds, n_unique_action)
            Logit values to define the evaluation policy.

        """

        expected_reward_click = self.fixed_expected_reward_click[user_idx]
        expected_reward_conversion = self.fixed_expected_reward_conversion[user_idx]

        check_array(array=context, name="context", expected_dim=2)
        check_array(
            array=evaluation_policy_logit_,
            name="evaluation_policy_logit_",
            expected_dim=2,
        )
        if evaluation_policy_logit_.shape[1] != self.n_unique_action:
            raise ValueError(
                "Expected `evaluation_policy_logit_.shape[1] != self.n_unique_action`,"
                "but found it False"
            )
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )
        if evaluation_policy_logit_.shape[0] != context.shape[0]:
            raise ValueError(
                "Expected `evaluation_policy_logit_.shape[0] == context.shape[0]`,"
                "but found it False"
            )

        if self.is_factorizable:
            enumerated_slate_actions = [
                _
                for _ in product(np.arange(self.n_unique_action), repeat=self.len_list)
            ]
        else:
            enumerated_slate_actions = [
                _ for _ in permutations(np.arange(self.n_unique_action), self.len_list)
            ]
        enumerated_slate_actions = np.array(enumerated_slate_actions).astype("int8")
        n_slate_actions = len(enumerated_slate_actions)
        n_rounds = len(evaluation_policy_logit_)

        pscores = []
        n_enumerated_slate_actions = len(enumerated_slate_actions)
        if self.is_factorizable:
            for action_list in tqdm(
                enumerated_slate_actions,
                desc="[calc_ground_truth_policy_value (pscore)]",
                total=n_enumerated_slate_actions,
            ):
                # for action_list in enumerated_slate_actions:
                pscores.append(
                    gen_eps_greedy(evaluation_policy_logit_, eps=eps)[
                        :, action_list
                    ].prod(1)
                )
            pscores = np.array(pscores).T
        else:
            for i in tqdm(
                np.arange(n_rounds),
                desc="[calc_ground_truth_policy_value (pscore)]",
                total=n_rounds,
            ):
                # for i in np.arange(n_rounds):
                pscores.append(
                    self._calc_pscore_given_policy_logit_epsilon_greedy(
                        all_slate_actions=enumerated_slate_actions,
                        policy_logit_i_=evaluation_policy_logit_[i],
                        eps=eps,
                    )
                )
            pscores = np.array(pscores)

        # calculate expected slate-level reward for each combinatorial set of items (i.e., slate actions)
        if self.base_reward_function is None:
            expected_slot_reward = self.sample_contextfree_expected_reward(
                random_state=self.random_state
            )
            expected_slot_reward_tile = np.tile(
                expected_slot_reward, (n_rounds * n_slate_actions, 1, 1)
            )
            expected_slate_rewards = np.array(
                [
                    expected_slot_reward_tile[
                        np.arange(n_slate_actions) % n_slate_actions,
                        np.array(enumerated_slate_actions)[:, pos_],
                        pos_,
                    ]
                    for pos_ in np.arange(self.len_list)
                ]
            ).T
            policy_value = (pscores * expected_slate_rewards.sum(axis=1)).sum()
        else:
            n_batch = (
                n_rounds * n_enumerated_slate_actions * self.len_list - 1
            ) // 10**7 + 1
            batch_size = (n_rounds - 1) // n_batch + 1
            n_batch = (n_rounds - 1) // batch_size + 1

            policy_value = 0.0
            for batch_idx in tqdm(
                np.arange(n_batch),
                desc=f"[calc_ground_truth_policy_value (expected reward), batch_size={batch_size}]",
                total=n_batch,
            ):
                # for batch_idx in np.arange(n_batch):
                context_ = context[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]
                pscores_ = pscores[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]

                expected_reward_click_ = expected_reward_click[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]
                expected_reward_conversion_ = expected_reward_conversion[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]

                # expected_slate_rewards_ = self.reward_function(
                #     context=context_,
                #     action_context=self.action_context,
                #     action=enumerated_slate_actions.flatten(),
                #     action_interaction_weight_matrix=self.action_interaction_weight_matrix,
                #     base_reward_function=self.base_reward_function,
                #     reward_type=self.reward_type,
                #     reward_structure=self.reward_structure,
                #     len_list=self.len_list,
                #     is_enumerated=True,
                #     random_state=self.random_state,
                # )

                expected_reward_factual_click = self.reward_function(
                    context=context_,
                    action_context=self.action_context,
                    action=enumerated_slate_actions.flatten(),
                    action_interaction_weight_matrix=self.action_interaction_weight_matrix,
                    expected_reward=expected_reward_click_,
                    reward_type=self.reward_type,
                    reward_structure=self.reward_structure,
                    len_list=self.len_list,
                    is_enumerated=True,
                    random_state=self.random_state,
                )
                expected_reward_factual_conversion = action_interaction_reward_function_conversion(
                    context=context_,
                    action_context=self.action_context,
                    action=enumerated_slate_actions.flatten(),
                    action_interaction_weight_matrix=self.action_interaction_weight_matrix_conversion,
                    expected_reward=expected_reward_conversion_,
                    reward_type=self.reward_type_conversion,
                    reward_structure=self.reward_structure_conversion,
                    len_list=self.len_list,
                    is_enumerated=True,
                    random_state=self.random_state + 555,
                    effect_from_ranking=self.effect_from_ranking,
                )
                expected_slate_rewards_ = (
                    expected_reward_factual_click * expected_reward_factual_conversion
                )

                # # click models based on expected reward
                # expected_slate_rewards_ *= self.exam_weight
                # if self.reward_type == "binary":
                #     discount_factors = np.ones(expected_slate_rewards_.shape[0])
                #     previous_slot_expected_reward = np.zeros(
                #         expected_slate_rewards_.shape[0]
                #     )
                #     for pos_ in np.arange(self.len_list):
                #         discount_factors *= (
                #             previous_slot_expected_reward * self.attractiveness[pos_]
                #             + (1 - previous_slot_expected_reward)
                #         )
                #         expected_slate_rewards_[:, pos_] = (
                #             discount_factors * expected_slate_rewards_[:, pos_]
                #         )
                #         previous_slot_expected_reward = expected_slate_rewards_[:, pos_]

                policy_value += (
                    pscores_.flatten() * expected_slate_rewards_.sum(axis=1)
                ).sum()
            policy_value /= n_rounds

        return policy_value

    def _calc_pscore_given_policy_logit_epsilon_greedy(
        self,
        all_slate_actions: np.ndarray,
        policy_logit_i_: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Calculate the propensity score of all possible slate actions given a particular policy_logit.

        Parameters
        ------------
        all_slate_actions: array-like, (n_action, len_list)
            All possible slate actions.

        policy_logit_i_: array-like, (n_unique_action, )
            Logit values given context (:math:`x`), which defines the distribution over actions of the policy.

        Returns
        ------------
        pscores: array-like, (n_action, )
            Propensity scores of all slate actions.

        """
        n_actions = len(all_slate_actions)
        unique_action_set_2d = np.tile(np.arange(self.n_unique_action), (n_actions, 1))
        pscores = np.ones(n_actions)
        for pos_ in np.arange(self.len_list):
            action_index = np.where(
                unique_action_set_2d == all_slate_actions[:, pos_][:, np.newaxis]
            )[1]
            pscores *= gen_eps_greedy(policy_logit_i_[unique_action_set_2d], eps=eps)[
                np.arange(n_actions), action_index
            ]
            # delete actions
            if pos_ + 1 != self.len_list:
                mask = np.ones((n_actions, self.n_unique_action - pos_))
                mask[np.arange(n_actions), action_index] = 0
                unique_action_set_2d = unique_action_set_2d[mask.astype(bool)].reshape(
                    (-1, self.n_unique_action - pos_ - 1)
                )

        return pscores

    def obtain_pscore_given_evaluation_policy_logit_epsilon_greedy(
        self,
        context: np.ndarray,
        action: np.ndarray,
        evaluation_policy_logit_: np.ndarray,
        return_pscore_item_position: bool = True,
        clip_logit_value: Optional[float] = None,
        eps: float = 0.2,
    ):
        """Calculate the propensity score given particular logit values to define the evaluation policy.

        Parameters
        ------------
        action: array-like, (n_rounds * len_list, )
            Action chosen by the behavior policy.

        evaluation_policy_logit_: array-like, (n_rounds, n_unique_action)
            Logit values to define the evaluation policy.

        return_pscore_item_position: bool, default=True
            Whether to compute `pscore_item_position` and include it in the logged data.
            When `n_actions` and `len_list` are large, `return_pscore_item_position`=True can lead to a long computation time.

        clip_logit_value: Optional[float], default=None
            A float parameter used to clip logit values (<= `700.`).
            When None, clipping is not applied to softmax values when obtaining `pscore_item_position`.
            When a float value is given, logit values are clipped when calculating softmax values.
            When `n_actions` and `len_list` are large, `clip_logit_value`=None can lead to a long computation time.

        """
        check_array(array=action, name="action", expected_dim=1)
        check_array(
            array=evaluation_policy_logit_,
            name="evaluation_policy_logit_",
            expected_dim=2,
        )
        if (
            len(action) / self.len_list != len(evaluation_policy_logit_)
            or evaluation_policy_logit_.shape[1] != self.n_unique_action
        ):
            raise ValueError(
                "the shape of `action` and `evaluation_policy_logit_` must be (n_rounds * len_list, )"
                "and (n_rounds, n_unique_action) respectively"
            )

        n_rounds = action.reshape((-1, self.len_list)).shape[0]
        pscore_cascade = np.zeros(n_rounds * self.len_list)
        pscore = np.zeros(n_rounds * self.len_list)
        p_click = np.zeros((n_rounds, self.n_unique_action))
        p_click_factual = np.zeros(n_rounds * self.len_list)
        if return_pscore_item_position:
            pscore_item_position = np.zeros(n_rounds * self.len_list)
            if not self.is_factorizable:
                enumerated_slate_actions = [
                    _
                    for _ in permutations(
                        np.arange(self.n_unique_action), self.len_list
                    )
                ]
                enumerated_slate_actions = np.array(enumerated_slate_actions)
        else:
            pscore_item_position = None
        if return_pscore_item_position and clip_logit_value is not None:
            check_scalar(
                clip_logit_value,
                name="clip_logit_value",
                target_type=(float),
                max_val=700.0,
            )
            evaluation_policy_softmax_ = np.exp(
                np.minimum(evaluation_policy_logit_, clip_logit_value)
            )
        # for i in tqdm(
        #     np.arange(n_rounds),
        #     desc="[obtain_pscore_given_evaluation_policy_logit]",
        #     total=n_rounds,
        # ):
        for i in np.arange(n_rounds):
            unique_action_set = np.arange(self.n_unique_action)
            score_ = gen_eps_greedy(
                evaluation_policy_logit_[i : i + 1, unique_action_set], eps=eps
            )[0]
            pscore_i = 1.0
            pscores = 0
            action_for_i = np.zeros(self.len_list)
            for pos_ in np.arange(self.len_list):
                action_ = action[i * self.len_list + pos_]
                action_index_ = np.where(unique_action_set == action_)[0][0]
                action_for_i[pos_] = action_
                # calculate joint pscore
                # print(score_)
                pscore_i *= score_[action_index_]
                pscore_cascade[i * self.len_list + pos_] = pscore_i
                # update the pscore given the remaining items for nonfactorizable policy
                if not self.is_factorizable and pos_ != self.len_list - 1:
                    unique_action_set = np.delete(
                        unique_action_set, unique_action_set == action_
                    )
                    score_ = gen_eps_greedy(
                        evaluation_policy_logit_[i : i + 1, unique_action_set],
                        eps=eps,
                    )[0]
                # calculate pscore_item_position
                if return_pscore_item_position:
                    if pos_ == 0:
                        pscore_item_pos_i_l = pscore_i
                    elif self.is_factorizable:
                        pscore_item_pos_i_l = score_[action_index_]
                    else:
                        if isinstance(clip_logit_value, float):
                            pscores = self._calc_pscore_given_policy_softmax(
                                all_slate_actions=enumerated_slate_actions,
                                policy_softmax_i_=evaluation_policy_softmax_[i],
                            )
                        else:
                            if pos_ == 1:
                                pscores = (
                                    self._calc_pscore_given_policy_logit_epsilon_greedy(
                                        all_slate_actions=enumerated_slate_actions,
                                        policy_logit_i_=evaluation_policy_logit_[i],
                                        eps=eps,
                                    )
                                )
                            else:
                                assert pscores is not None, (
                                    "pscores should have been calculated at pos_ == 0"
                                )

                        pscore_item_pos_i_l = pscores[
                            enumerated_slate_actions[:, pos_] == action_
                        ].sum()
                    pscore_item_position[i * self.len_list + pos_] = pscore_item_pos_i_l
            expected_reward_all_click = self.reward_function(
                context=context[i].reshape(1, -1),
                action_context=self.action_context,
                action=enumerated_slate_actions.flatten(),
                action_interaction_weight_matrix=self.action_interaction_weight_matrix,
                expected_reward=self.expected_reward_click,
                base_reward_function=self.base_reward_function,
                reward_type=self.reward_type,
                reward_structure=self.reward_structure,
                len_list=self.len_list,
                is_enumerated=True,
                random_state=self.random_state,
            )
            # p_click for all unique action
            for a in np.arange(self.n_unique_action):
                idx = np.where(enumerated_slate_actions == a)
                p_A = pscores[idx[0]]
                q_x_c = expected_reward_all_click[idx]
                p_click[i, a] = (p_A * q_x_c).sum()
            # p_click_factual[i*self.len_list:i*self.len_list+self.len_list] = action_for_i
            p_click_factual[i * self.len_list : i * self.len_list + self.len_list] = (
                p_click[i, action_for_i.astype(int)]
            )

            # impute joint pscore
            start_idx = i * self.len_list
            end_idx = start_idx + self.len_list
            pscore[start_idx:end_idx] = pscore_i
        return pscore, pscore_item_position, pscore_cascade, p_click_factual, p_click

    def obtain_p_click_pi_given_estimated_click_probability(
        self,
        context: np.ndarray,
        action: np.ndarray,
        click_model: np.ndarray,
        evaluation_policy_logit_type: str,
        eps: float = 0.2,
        tau: float = 1.0,
    ):

        if evaluation_policy_logit_type == "linear_reward_function":
            evaluation_policy_logit_ = linear_reward_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        else:
            evaluation_policy_logit_ = linear_behavior_policy_logit(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
                tau=tau,
            )

        action = action.reshape((-1, self.len_list))
        enumerated_slate_actions = [
            _ for _ in permutations(np.arange(self.n_unique_action), self.len_list)
        ]
        enumerated_slate_actions = np.array(enumerated_slate_actions)

        behavior_policy_logit_ = self.behavior_policy_function(
            context=context,
            action_context=self.action_context,
            random_state=self.random_state,
        )

        n_rounds = context.shape[0]
        p_click_pi_0 = np.zeros((n_rounds, self.n_unique_action))
        p_click_pi_e = np.zeros((n_rounds, self.n_unique_action))
        p_click_factual_pi_0 = np.zeros(n_rounds * self.len_list)
        p_click_factual_pi_e = np.zeros(n_rounds * self.len_list)

        for i in np.arange(n_rounds):
            is_deterministic = False
            if context[i][0] <= self.deterministic_user_threshold:
                is_deterministic = True

            pscores_pi_0 = self._calc_pscore_given_policy_logit(
                all_slate_actions=enumerated_slate_actions,
                policy_logit_i_=behavior_policy_logit_[i],
                is_deterministic=is_deterministic,
            )

            pscores_pi_e = self._calc_pscore_given_policy_logit_epsilon_greedy(
                all_slate_actions=enumerated_slate_actions,
                policy_logit_i_=evaluation_policy_logit_[i],
                eps=eps,
            )

            input_context = np.repeat(
                context[i].reshape(1, -1), enumerated_slate_actions.shape[0], axis=0
            )
            X_test = np.concatenate([input_context, enumerated_slate_actions], axis=1)
            estimated_click_probability_i = click_model.predict_proba(X_test)

            # p_click for all unique action
            for a in np.arange(self.n_unique_action):
                idx = np.where(enumerated_slate_actions == a)
                q_x_c = estimated_click_probability_i[idx]

                pi_0_A_x = pscores_pi_0[idx[0]]
                pi_e_A_x = pscores_pi_e[idx[0]]

                p_click_pi_0[i, a] = (pi_0_A_x * q_x_c).sum()
                p_click_pi_e[i, a] = (pi_e_A_x * q_x_c).sum()

            p_click_factual_pi_0[
                i * self.len_list : i * self.len_list + self.len_list
            ] = p_click_pi_0[i, action[i]]
            p_click_factual_pi_e[
                i * self.len_list : i * self.len_list + self.len_list
            ] = p_click_pi_e[i, action[i]]

        return p_click_factual_pi_0, p_click_factual_pi_e, p_click_pi_e


###########################################################################################
def generate_symmetric_matrix(n_unique_action: int, random_state: int) -> np.ndarray:
    """Generate symmetric matrix

    Parameters
    -----------
    n_unique_action: int (>= len_list)
        Number of unique actions.

    random_state: int
        Controls the random seed in sampling elements of matrix.

    Returns
    ---------
    symmetric_matrix: array-like, shape (n_unique_action, n_unique_action)

    """
    random_ = check_random_state(random_state)
    # base_matrix = random_.normal(scale=0.5, size=(n_unique_action, n_unique_action))
    base_matrix = random_.uniform(
        low=-3.0, high=3.0, size=(n_unique_action, n_unique_action)
    )
    symmetric_matrix = (
        np.tril(base_matrix) + np.tril(base_matrix).T - np.diag(base_matrix.diagonal())
    )
    return symmetric_matrix


def generate_symmetric_matrix_conversion(
    n_unique_action: int, random_state: int
) -> np.ndarray:
    """Generate symmetric matrix

    Parameters
    -----------
    n_unique_action: int (>= len_list)
        Number of unique actions.

    random_state: int
        Controls the random seed in sampling elements of matrix.

    Returns
    ---------
    symmetric_matrix: array-like, shape (n_unique_action, n_unique_action)

    """
    random_ = check_random_state(random_state)
    # base_matrix = random_.normal(scale=0.5, size=(n_unique_action, n_unique_action))
    base_matrix = random_.uniform(
        low=-1.0, high=1.0, size=(n_unique_action, n_unique_action)
    )
    symmetric_matrix = (
        np.tril(base_matrix) + np.tril(base_matrix).T - np.diag(base_matrix.diagonal())
    )
    # print(symmetric_matrix)
    return symmetric_matrix


def action_interaction_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    action: np.ndarray,
    expected_reward: np.ndarray,
    reward_type: str,
    reward_structure: str,
    action_interaction_weight_matrix: np.ndarray,
    len_list: int,
    is_enumerated: bool = False,
    random_state: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Reward function incorporating interactions among combinatorial action

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_unique_action, dim_action_context)
        Vector representation of actions.

    action: array-like, shape (n_rounds * len_list, ) or (len(enumerated_slate_actions) * len_list, )
        When `is_enumerated`=False, action corresponds to actions sampled by a (often behavior) policy.
        In this case, actions sampled within slate `i` is stored in `action[`i` * `len_list`: (`i + 1`) * `len_list`]`.
        When `is_enumerated`=True, action corresponds to the enumerated all possible combinatorial actions.

    base_reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function to define the expected reward, i.e., :math:`q: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.

    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary',the expected rewards are transformed by logit function.

    reward_structure: str
        Reward structure.
        Must be one of 'standard_additive', 'cascade_additive', 'standard_decay', or 'cascade_decay'.

    action_interaction_weight_matrix (`W`): array-like, shape (n_unique_action, n_unique_action) or (len_list, len_list)
        When using an additive-type reward_structure, `W(i, j)` defines the interaction between action `i` and `j`.
        When using an decay-type reward_structure, `W(i, j)` defines the weight of how the expected reward of slot `i` affects that of slot `j`.
        See the experiment section of Kiyohara et al.(2022) for details.

    len_list: int (> 1)
        Length of a list/ranking of actions, slate size.

    is_enumerate: bool
        Whether `action` corresponds to `enumerated_slate_actions`.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward_factual: array-like, shape (n_rounds, len_list)
        When reward_structure='standard_additive', :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j \\neq k} W(a(k), a(j)))`.
        When reward_structure='cascade_additive', :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j < k} W(a(k), a(j)))`.
        Otherwise, :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j \\neq k} g^{-1}(f(x, a(j))) * W(k, j)`

    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)
    check_array(array=action, name="action", expected_dim=1)
    if is_enumerated and action.shape[0] % len_list != 0:
        raise ValueError(
            "Expected `action.shape[0] % len_list == 0` if `is_enumerated is True`,"
            "but found it False"
        )
    if not is_enumerated and action.shape[0] != len_list * context.shape[0]:
        raise ValueError(
            "Expected `action.shape[0] == len_list * context.shape[0]` if `is_enumerated is False`, but found it False"
        )
    if reward_type not in [
        "binary",
        "continuous",
    ]:
        raise ValueError(
            f"`reward_type` must be either 'binary' or 'continuous', but {reward_type} is given."
        )
    if reward_structure not in [
        "standard_additive",
        "cascade_additive",
        "standard_decay",
        "cascade_decay",
        "independent",
    ]:
        raise ValueError(
            f"`reward_structure` must be either 'standard_additive', 'cascade_additive', 'standard_decay' or 'cascade_decay', but {reward_structure} is given."
        )

    is_additive = reward_structure in ["standard_additive", "cascade_additive"]
    is_cascade = reward_structure in ["cascade_additive", "cascade_decay"]

    if is_additive:
        if action_interaction_weight_matrix.shape != (
            action_context.shape[0],
            action_context.shape[0],
        ):
            raise ValueError(
                f"the shape of `action_interaction_weight_matrix` must be `(action_context.shape[0], action_context.shape[0])`, but {action_interaction_weight_matrix.shape}"
            )
    else:  # decay
        if action_interaction_weight_matrix.shape != (
            len_list,
            len_list,
        ):
            raise ValueError(
                f"the shape of `action_interaction_weight_matrix` must be `(len_list, len_list)`, but {action_interaction_weight_matrix.shape}"
            )

    n_rounds = context.shape[0]
    # duplicate action
    if is_enumerated:
        action = np.tile(action, n_rounds)
    # action_2d: array-like, shape (n_rounds (* len(enumerated_action_list)), len_list)
    action_2d = action.reshape((-1, len_list)).astype("int8")
    n_enumerated_slate_actions = len(action) // n_rounds
    # expected_reward: array-like, shape (n_rounds, n_unique_action)
    # expected_reward = base_reward_function(
    #     context=context, action_context=action_context, random_state=random_state
    # )
    # if reward_type == "binary":
    #     expected_reward = logit(expected_reward)
    expected_reward_factual = np.zeros_like(action_2d, dtype="float16")
    for pos_ in np.arange(len_list):
        tmp_fixed_reward = expected_reward[
            np.arange(len(action_2d)) // n_enumerated_slate_actions,
            action_2d[:, pos_],
        ]
        if reward_structure == "independent":
            pass
        elif is_additive:
            for pos2_ in np.arange(len_list):
                if is_cascade:
                    if pos_ <= pos2_:
                        break
                elif pos_ == pos2_:
                    continue
                distance = 1 / np.abs(pos_ - pos2_)
                tmp_fixed_reward += (
                    distance
                    * action_interaction_weight_matrix[
                        action_2d[:, pos_], action_2d[:, pos2_]
                    ]
                )
        else:
            for pos2_ in np.arange(len_list):
                if is_cascade:
                    if pos_ <= pos2_:
                        break
                elif pos_ == pos2_:
                    continue
                expected_reward_ = expected_reward[
                    np.arange(len(action_2d)) // n_enumerated_slate_actions,
                    action_2d[:, pos2_],
                ]
                weight_ = action_interaction_weight_matrix[pos_, pos2_]
                tmp_fixed_reward += expected_reward_ * weight_
        expected_reward_factual[:, pos_] = tmp_fixed_reward

    if reward_type == "binary":
        # expected_reward_factual /= len_list
        expected_reward_factual[:, :] = sigmoid(expected_reward_factual) / (
            np.arange(len_list) + 1
        )
    else:
        expected_reward_factual /= np.arange(len_list) + 1
        # expected_reward_factual /= len_list
        # expected_reward_factual = np.clip(expected_reward_factual, 0, None)

    assert expected_reward_factual.shape == (
        action_2d.shape[0],
        len_list,
    ), (
        f"response shape must be (n_rounds (* enumerated_slate_actions), len_list), but {expected_reward_factual.shape}"
    )
    return expected_reward_factual


def action_interaction_reward_function_conversion(
    context: np.ndarray,
    action_context: np.ndarray,
    action: np.ndarray,
    expected_reward: np.ndarray,
    reward_type: str,
    reward_structure: str,
    action_interaction_weight_matrix: np.ndarray,
    len_list: int,
    is_enumerated: bool = False,
    random_state: Optional[int] = None,
    effect_from_ranking: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """Reward function incorporating interactions among combinatorial action

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_unique_action, dim_action_context)
        Vector representation of actions.

    action: array-like, shape (n_rounds * len_list, ) or (len(enumerated_slate_actions) * len_list, )
        When `is_enumerated`=False, action corresponds to actions sampled by a (often behavior) policy.
        In this case, actions sampled within slate `i` is stored in `action[`i` * `len_list`: (`i + 1`) * `len_list`]`.
        When `is_enumerated`=True, action corresponds to the enumerated all possible combinatorial actions.

    base_reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function to define the expected reward, i.e., :math:`q: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.

    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary',the expected rewards are transformed by logit function.

    reward_structure: str
        Reward structure.
        Must be one of 'standard_additive', 'cascade_additive', 'standard_decay', or 'cascade_decay'.

    action_interaction_weight_matrix (`W`): array-like, shape (n_unique_action, n_unique_action) or (len_list, len_list)
        When using an additive-type reward_structure, `W(i, j)` defines the interaction between action `i` and `j`.
        When using an decay-type reward_structure, `W(i, j)` defines the weight of how the expected reward of slot `i` affects that of slot `j`.
        See the experiment section of Kiyohara et al.(2022) for details.

    len_list: int (> 1)
        Length of a list/ranking of actions, slate size.

    is_enumerate: bool
        Whether `action` corresponds to `enumerated_slate_actions`.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward_factual: array-like, shape (n_rounds, len_list)
        When reward_structure='standard_additive', :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j \\neq k} W(a(k), a(j)))`.
        When reward_structure='cascade_additive', :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j < k} W(a(k), a(j)))`.
        Otherwise, :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j \\neq k} g^{-1}(f(x, a(j))) * W(k, j)`

    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)
    check_array(array=action, name="action", expected_dim=1)
    if is_enumerated and action.shape[0] % len_list != 0:
        raise ValueError(
            "Expected `action.shape[0] % len_list == 0` if `is_enumerated is True`,"
            "but found it False"
        )
    if not is_enumerated and action.shape[0] != len_list * context.shape[0]:
        raise ValueError(
            "Expected `action.shape[0] == len_list * context.shape[0]` if `is_enumerated is False`, but found it False"
        )
    if reward_type not in [
        "binary",
        "continuous",
    ]:
        raise ValueError(
            f"`reward_type` must be either 'binary' or 'continuous', but {reward_type} is given."
        )
    if reward_structure not in [
        "standard_additive",
        "cascade_additive",
        "standard_decay",
        "cascade_decay",
        "independent",
    ]:
        raise ValueError(
            f"`reward_structure` must be either 'standard_additive', 'cascade_additive', 'standard_decay' or 'cascade_decay', but {reward_structure} is given."
        )

    is_additive = reward_structure in ["standard_additive", "cascade_additive"]
    is_cascade = reward_structure in ["cascade_additive", "cascade_decay"]

    if is_additive:
        if action_interaction_weight_matrix.shape != (
            action_context.shape[0],
            action_context.shape[0],
        ):
            raise ValueError(
                f"the shape of `action_interaction_weight_matrix` must be `(action_context.shape[0], action_context.shape[0])`, but {action_interaction_weight_matrix.shape}"
            )
    else:  # decay
        if action_interaction_weight_matrix.shape != (
            len_list,
            len_list,
        ):
            raise ValueError(
                f"the shape of `action_interaction_weight_matrix` must be `(len_list, len_list)`, but {action_interaction_weight_matrix.shape}"
            )

    n_rounds = context.shape[0]

    # duplicate action
    if is_enumerated:
        action = np.tile(action, n_rounds)
    # action_2d: array-like, shape (n_rounds (* len(enumerated_action_list)), len_list)
    action_2d = action.reshape((-1, len_list)).astype("int8")
    n_enumerated_slate_actions = len(action) // n_rounds
    # expected_reward: array-like, shape (n_rounds, n_unique_action)
    # expected_reward = base_reward_function(
    #     context=context, action_context=action_context, random_state=random_state
    # )
    # print(expected_reward)
    if reward_type == "binary":
        # expected_reward = np.abs(expected_reward)
        expected_reward = logit(expected_reward)
    expected_reward_factual = np.zeros_like(action_2d, dtype="float16")
    # print(action.shape)
    # print(expected_reward.shape)
    for pos_ in np.arange(len_list):
        # print(np.arange(len(action_2d)),action_2d[:, pos_])
        # print(expected_reward.shape)
        # print(action_2d.shape)
        # print((np.arange(len(action_2d)) // n_enumerated_slate_actions).shape)
        # print(np.arange(len(action_2d)) // n_enumerated_slate_actions)

        tmp_fixed_reward = expected_reward[
            np.arange(len(action_2d)) // n_enumerated_slate_actions,
            action_2d[:, pos_],
        ]
        # print("-------------------",len(action_2d))
        # print(np.arange(len(action_2d)) // n_enumerated_slate_actions,
        #     action_2d[:, pos_])
        # print(expected_reward[np.arange(10),action_2d[:, pos_]])
        # print(tmp_fixed_reward)

        if reward_structure == "independent":
            pass
        elif is_additive:
            for pos2_ in np.arange(len_list):
                if is_cascade:
                    if pos_ <= pos2_:
                        break
                elif pos_ == pos2_:
                    continue
                distance = 1 / np.abs(pos_ - pos2_)
                tmp_fixed_reward += (
                    effect_from_ranking
                    * distance
                    * action_interaction_weight_matrix[
                        action_2d[:, pos_], action_2d[:, pos2_]
                    ]
                )
        else:
            for pos2_ in np.arange(len_list):
                if is_cascade:
                    if pos_ <= pos2_:
                        break
                elif pos_ == pos2_:
                    continue
                expected_reward_ = expected_reward[
                    np.arange(len(action_2d)) // n_enumerated_slate_actions,
                    action_2d[:, pos2_],
                ]
                weight_ = action_interaction_weight_matrix[pos_, pos2_]
                tmp_fixed_reward += expected_reward_ * weight_
        expected_reward_factual[:, pos_] = tmp_fixed_reward

    if reward_type == "binary":
        # expected_reward_factual /= len_list
        expected_reward_factual[:, :] = sigmoid(expected_reward_factual)
    else:
        expected_reward_factual *= 1
        # expected_reward_factual /= len_list
        # expected_reward_factual = np.clip(expected_reward_factual, 0, None)

    assert expected_reward_factual.shape == (
        action_2d.shape[0],
        len_list,
    ), (
        f"response shape must be (n_rounds (* enumerated_slate_actions), len_list), but {expected_reward_factual.shape}"
    )

    return expected_reward_factual


def linear_behavior_policy_logit(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
    tau: Union[int, float] = 1.0,
) -> np.ndarray:
    """Linear contextual behavior policy for synthetic slate bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_unique_action, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    tau: int or float, default=1.0
        A temperature parameter to control the entropy of the behavior policy.
        As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.

    Returns
    ---------
    logit value: array-like, shape (n_rounds, n_unique_action)
        Logit values to define the behavior policy.

    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)
    # check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

    random_ = check_random_state(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    coef_ = random_.uniform(size=context.shape[1])
    action_coef_ = random_.uniform(size=action_context.shape[1])
    context_action_coef = random_.uniform(
        size=(context.shape[0], action_context.shape[0])
    )
    for d in np.arange(action_context.shape[0]):
        logits[:, d] = context @ coef_ + action_context[d] @ action_coef_
        logits[:, d] += context_action_coef[:, d]

    return logits / tau


def exponential_decay_function(distance: np.ndarray) -> np.ndarray:
    """Calculate exponential discount factor for action interaction weight matrix.

    Parameters
    -----------
    distance: array-like, shape (len_list, )
        Distance between two slots.

    """
    check_array(array=distance, name="distance", expected_dim=1)

    return np.exp(-distance)


def inverse_decay_function(distance: np.ndarray) -> np.ndarray:
    """Calculate inverse discount factor for action interaction weight matrix.

    Parameters
    -----------
    distance: array-like, shape (len_list, )
        Distance between two slots.

    """
    check_array(array=distance, name="distance", expected_dim=1)

    return 1 / (distance + 1)
