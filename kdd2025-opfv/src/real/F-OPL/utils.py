# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import datetime
from dataclasses import dataclass

import conf
import numpy as np
import pandas as pd
import torch
from sklearn.utils import check_random_state


def sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    sampled_actions = flg.argmax(axis=1)
    return sampled_actions


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


def softmax(x: np.ndarray) -> np.ndarray:
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


@dataclass
class RegBasedPolicyDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert self.context.shape[0] == self.action.shape[0] == self.reward.shape[0]

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class GradientBasedPolicyDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    q_hat: np.ndarray
    pi_0: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.q_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class Prognosticatordataset(torch.utils.data.Dataset):
    context: np.ndarray
    time: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    # q_hat: np.ndarray
    pi_0: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.time.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            # == self.q_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.time[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            # self.q_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class OPFVDataset(torch.utils.data.Dataset):
    context: np.ndarray
    time: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    q_hat: np.ndarray
    pi_0: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.time.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.q_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.time[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.context.shape[0]


def add_median_rank(result_df_list, df_stats_rank_list):
    for i in range(len(result_df_list)):
        result_df = result_df_list[i]
        df_stats_rank = df_stats_rank_list[i]
        ### Calculate the median of each column in the DataFrame of the results of the DM, SNIPS, SNDR evaluation
        median = result_df.median()

        ### Calculate the median of rank for each row in the DataFrame of the results of the DM, SNIPS, SNDR evaluation
        rank_each_row = result_df.rank(axis=1, ascending=False)
        median_rank_each_row = rank_each_row.median()

        ### Add New column on median and median of rank
        df_stats_rank["Median"] = median.to_list()
        df_stats_rank["Median of Rank"] = median_rank_each_row.to_list()


def create_result_df_stats(result_df):
    result_df_stats = pd.DataFrame({"mean": result_df.mean(), "std": result_df.std()})
    return result_df_stats


def create_rank_df_mean(result_df):
    rank_df = result_df.rank(axis=1, ascending=False)
    rank_df = rank_df.T
    rank_df_mean_rank = pd.DataFrame({"Mean Rank": rank_df.mean(axis=1)})
    return rank_df_mean_rank


def create_df_mean_std_rank(
    result_df_DM, result_df_IPS, result_df_SNIPS, result_df_SNDR, result_df_3_mean
):
    result_df_DM_stats = create_result_df_stats(result_df_DM)

    result_df_IPS_stats = create_result_df_stats(result_df_IPS)

    result_df_SNIPS_stats = create_result_df_stats(result_df_SNIPS)

    result_df_SNDR_stats = create_result_df_stats(result_df_SNDR)

    result_df_3_mean_stats = create_result_df_stats(result_df_3_mean)

    rank_df_DM_mean_rank = create_rank_df_mean(result_df_DM)

    rank_df_IPS_mean_rank = create_rank_df_mean(result_df_IPS)

    rank_df_SNIPS_mean_rank = create_rank_df_mean(result_df_SNIPS)

    rank_df_SNDR_mean_rank = create_rank_df_mean(result_df_SNDR)

    rank_df_3_mean_mean_rank = create_rank_df_mean(result_df_3_mean)

    return (
        result_df_DM_stats,
        result_df_IPS_stats,
        result_df_SNIPS_stats,
        result_df_SNDR_stats,
        result_df_3_mean_stats,
        rank_df_DM_mean_rank,
        rank_df_IPS_mean_rank,
        rank_df_SNIPS_mean_rank,
        rank_df_SNDR_mean_rank,
        rank_df_3_mean_mean_rank,
    )


@dataclass
class BehaviorPolicyDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert self.context.shape[0] == self.action.shape[0]

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
        )

    def __len__(self):
        return self.context.shape[0]


fromtimestamp_vec = np.vectorize(datetime.datetime.fromtimestamp)


def show_hyperparameters(
    time_at_evaluation_start: int = None,
    time_at_evaluation_end: int = None,
    flag_show_time_at_evaluation: bool = True,
    time_at_evaluation_list: callable = None,
):
    print(f"################# START hyperparameters #################")

    print(f"### About Seeds and Number of Samples ###")
    print(f"number of seeds = {conf.n_seeds}")
    print(f"number of training samples (n) = {conf.num_train}")
    print(f"number of test samples = {conf.num_test}\n")

    print(f"### About Time Structure ###")
    print(
        f"number of true time structures for reward (|C_r|) = {conf.num_time_structure_for_logged_data}"
    )
    print(f"strength of time structure for reward (lambda) = {conf.lambda_ratio}\n")

    print(f"### About OPL ###")
    print(f"number of epochs = {conf.max_iter}")
    print(f"batch size = {conf.batch_size}")
    print(
        f"number of the samples of time when we learn a policy for each batch = {conf.num_time_learn}\n"
    )

    print(f"### About Prognosticator ###")
    print(f"list of time features for Prognosticator = {conf.phi_scalar_func_list}")
    print(
        f"optimality of the data driven feature selection for Prognosticator = {conf.flag_Prognosticator_optimality}"
    )
    print(
        f"number of time features for Prognosticator = {conf.num_features_for_Prognosticator}"
    )
    print(
        f"list of the numbers of time features for Prognosticator = {conf.num_features_for_Prognosticator_list}\n"
    )

    print(f"### About Logged Data Collection Period and evaluation Period ###")
    print(
        f"time when we start collecting the logged data = {datetime.datetime.fromtimestamp(conf.t_oldest)}"
    )
    print(
        f"time when we finish collecting the logged data = {datetime.datetime.fromtimestamp(conf.t_now)}"
    )
    if flag_show_time_at_evaluation == True:
        print(
            f"time when we start evaluation a target policy = {datetime.datetime.fromtimestamp(time_at_evaluation_start)}"
        )
        print(
            f"time when we finish evaluation a target policy = {datetime.datetime.fromtimestamp(time_at_evaluation_end)}"
        )
    print(f"future time = {datetime.datetime.fromtimestamp(conf.t_future)}\n")

    print(f"### About Parameters for Data Generating Process ###")
    print(f"number of actions (|A|) = {conf.n_actions}")
    print(f"dimension of context (d_x) = {conf.dim_context}")
    print(f"number of users = {conf.n_users}")
    print(f"behavior policy optimality (beta) = {conf.beta}")
    print(f"target policy optimality (epsilon) = {conf.eps}\n")

    print(f"### About Varying Parameters ###")
    print(f"list of the numbers of training samples (n) = {conf.num_train_list}")
    print(
        f"list of the strengths of time structure for reward (lambda) = {conf.lambda_ratio_list}"
    )
    print(
        f"list of the numbers of candidate time structures for reward = {conf.candidate_num_time_structure_list}"
    )
    if flag_show_time_at_evaluation == False:
        print(
            f"list of the time at evaluation = {fromtimestamp_vec(time_at_evaluation_list)}"
        )

    print(f"################# END hyperparameters #################\n\n")
