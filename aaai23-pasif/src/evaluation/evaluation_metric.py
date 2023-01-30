# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.

import numpy as np
import pandas as pd
import copy
import statistics
from sklearn.metrics import precision_score
from scipy.stats import spearmanr



def calculate_relative_regret_e(true_data, estimated_data, estimator_selection_metrics='mse'):
    """calculate relative regret for evaluation of estimator selection

    Args:
        true_data (pd.DataFrame): dataframe including estimator name, estimator_selection_metric, and rank
        estimated_data (pd.DataFrame): dataframe including estimator name and rank
        estimator_selection_metrics (str, optional): Defaults to 'mse'. Must be 'mse' or 'mean relative-ee'

    Returns:
        float: relative regret
    """

    predicted_best_estimator_name = estimated_data['estimator_name'][estimated_data['rank']==1].values[0]
    true_best_estimator_name = true_data['estimator_name'][true_data['rank']==1].values[0]

    predicted_estimator_performance = true_data[estimator_selection_metrics][true_data['estimator_name']==predicted_best_estimator_name].values[0]
    true_estimator_performance = true_data[estimator_selection_metrics][true_data['estimator_name']==true_best_estimator_name].values[0]

    relative_regret_e = (predicted_estimator_performance / true_estimator_performance) - 1.0

    return relative_regret_e




def calculate_rank_correlation_coefficient_e(true_data, estimated_data):
    """calculate rank correlation coefficient for evaluation of estimator selection

    Args:
        true_data (pd.DataFrame): dataframe including estimator name and rank
        estimated_data (pd.DataFrame): dataframe including estimator name and rank

    Returns:
        float: rank correlation coefficient
    """

    merged_data = pd.merge(
        true_data.rename(columns={'rank':'rank_true'}), 
        estimated_data.rename(columns={'rank':'rank_predicted'}), 
        how='left', 
        on='estimator_name'
        )

    rank_true = merged_data['rank_true'].values
    rank_predict = merged_data['rank_predicted'].values
    rank_cc, pvalue = spearmanr(rank_predict, rank_true)

    return rank_cc





def calculate_relative_regret_p(true_data, estimated_data):
    """calculate relative regret for evaluation of policy selection

    Args:
        true_data (pd.DataFrame): dataframe including policy name, policy value, and rank
        estimated_data (pd.DataFrame): dataframe including policy name and rank

    Returns:
        float: relative regret
    """

    predicted_best_policy_name = estimated_data['policy_name'][estimated_data['rank']==1].values[0]
    true_best_policy_name = true_data['policy_name'][true_data['rank']==1].values[0]

    predicted_policy_performance = true_data['policy_value'][true_data['policy_name']==predicted_best_policy_name].values[0]
    true_policy_performance = true_data['policy_value'][true_data['policy_name']==true_best_policy_name].values[0]

    relative_regret_p = (predicted_policy_performance / true_policy_performance) - 1.0
    relative_regret_p = (-1.0) * relative_regret_p

    return relative_regret_p



def calculate_rank_correlation_coefficient_p(true_data, estimated_data):
    """calculate rank correlation coefficient for evaluation of policy selection

    Args:
        true_data (pd.DataFrame): dataframe including policy name and rank
        estimated_data (pd.DataFrame): dataframe including policy name and rank

    Returns:
        float: rank correlation coefficient
    """

    merged_data = pd.merge(
        true_data.rename(columns={'rank':'rank_true'}), 
        estimated_data.rename(columns={'rank':'rank_predicted'}), 
        how='left', 
        on='policy_name'
        )

    rank_true = merged_data['rank_true'].values
    rank_predict = merged_data['rank_predicted'].values
    rank_cc, pvalue = spearmanr(rank_predict, rank_true)

    return rank_cc

