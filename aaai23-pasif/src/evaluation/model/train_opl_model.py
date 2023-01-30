# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.

import numpy as np
import pandas as pd
import pickle
import argparse
import os
import sys

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, Ridge

import obp
from obp.policy import IPWLearner, QLearner, NNPolicyLearner, Random
from obp.dataset import SyntheticBanditDataset, logistic_reward_function, linear_reward_function, linear_behavior_policy


import warnings
warnings.simplefilter('ignore')



if __name__ == '__main__':
    
    # get argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_actions', type=int, default=5)
    parser.add_argument('--dim_context', type=int, default=5)
    parser.add_argument('--reward_function', type=int, choices=[0, 1], help='0:logistic_reward_function, 1:linear_reward_function', default=0)
    reward_function_dict = {0:logistic_reward_function, 1:linear_reward_function}
    parser.add_argument('--random_state', type=int, default=1)
    arguments = parser.parse_args()


    if os.path.dirname(__file__) == '':
        model_partial_path = './'
    else:
        model_partial_path = os.path.dirname(__file__) + '/'


    # case of binary-reward
    dataset_binary = SyntheticBanditDataset(
        n_actions=arguments.n_actions,
        dim_context=arguments.dim_context,
        beta=-10.0, # inverse temperature parameter to control the optimality and entropy of the behavior policy
        reward_type="binary", # "binary" or "continuous"
        reward_function=reward_function_dict[arguments.reward_function],
        random_state=arguments.random_state
        )
    bandit_feedback_binary_train = dataset_binary.obtain_batch_bandit_feedback(n_rounds=10000)
    ipw_lr = IPWLearner(
        n_actions=dataset_binary.n_actions,
        base_classifier=LogisticRegression(random_state=arguments.random_state)
        )
    ipw_lr.fit(
        context=bandit_feedback_binary_train["context"],
        action=bandit_feedback_binary_train["action"],
        reward=bandit_feedback_binary_train["reward"],
        pscore=bandit_feedback_binary_train["pscore"]
        )
    with open(model_partial_path+'ipw_lr_b.pickle', 'wb') as f:
        pickle.dump(ipw_lr, f)

    dataset_binary = SyntheticBanditDataset(
        n_actions=arguments.n_actions,
        dim_context=arguments.dim_context,
        beta=2.0, # inverse temperature parameter to control the optimality and entropy of the behavior policy
        reward_type="binary", # "binary" or "continuous"
        reward_function=reward_function_dict[arguments.reward_function],
        random_state=arguments.random_state
        )
    bandit_feedback_binary_train = dataset_binary.obtain_batch_bandit_feedback(n_rounds=10000)
    ipw_rf = IPWLearner(
        n_actions=dataset_binary.n_actions,
        base_classifier=RandomForestClassifier(n_jobs=-1, random_state=arguments.random_state)
        )
    ipw_rf.fit(
        context=bandit_feedback_binary_train["context"],
        action=bandit_feedback_binary_train["action"],
        reward=bandit_feedback_binary_train["reward"],
        pscore=bandit_feedback_binary_train["pscore"]
        )
    with open(model_partial_path+'ipw_rf_b.pickle', 'wb') as f:
        pickle.dump(ipw_rf, f)

    dataset_binary = SyntheticBanditDataset(
        n_actions=arguments.n_actions,
        dim_context=arguments.dim_context,
        beta=10.0, # inverse temperature parameter to control the optimality and entropy of the behavior policy
        reward_type="binary", # "binary" or "continuous"
        reward_function=reward_function_dict[arguments.reward_function],
        random_state=arguments.random_state
        )
    bandit_feedback_binary_train = dataset_binary.obtain_batch_bandit_feedback(n_rounds=10000)
    qlr_lr = QLearner(
        n_actions=dataset_binary.n_actions, 
        base_model=LogisticRegression(random_state=arguments.random_state)
        )
    qlr_lr.fit(
        context=bandit_feedback_binary_train["context"],
        action=bandit_feedback_binary_train["action"],
        reward=bandit_feedback_binary_train["reward"]
        )
    with open(model_partial_path+'qlr_lr_b.pickle', 'wb') as f:
        pickle.dump(qlr_lr, f)

    dataset_binary = SyntheticBanditDataset(
        n_actions=arguments.n_actions,
        dim_context=arguments.dim_context,
        beta=-2.0, # inverse temperature parameter to control the optimality and entropy of the behavior policy
        reward_type="binary", # "binary" or "continuous"
        reward_function=reward_function_dict[arguments.reward_function],
        random_state=arguments.random_state
        )
    bandit_feedback_binary_train = dataset_binary.obtain_batch_bandit_feedback(n_rounds=10000)
    qlr_rf = QLearner(
        n_actions=dataset_binary.n_actions, 
        base_model=RandomForestClassifier(n_jobs=-1, random_state=arguments.random_state)
        )
    qlr_rf.fit(
        context=bandit_feedback_binary_train["context"],
        action=bandit_feedback_binary_train["action"],
        reward=bandit_feedback_binary_train["reward"]
        )
    with open(model_partial_path+'qlr_rf_b.pickle', 'wb') as f:
        pickle.dump(qlr_rf, f)




    # case of continuous-reward
    dataset_continuous = SyntheticBanditDataset(
        n_actions=arguments.n_actions,
        dim_context=arguments.dim_context,
        beta=-10.0, # inverse temperature parameter to control the optimality and entropy of the behavior policy
        reward_type="continuous", # "binary" or "continuous"
        reward_function=reward_function_dict[arguments.reward_function],
        random_state=arguments.random_state
        )
    bandit_feedback_continuous_train = dataset_continuous.obtain_batch_bandit_feedback(n_rounds=10000)
    ipw_lr = IPWLearner(
        n_actions=dataset_continuous.n_actions,
        base_classifier=LogisticRegression(random_state=arguments.random_state)
        )
    ipw_lr.fit(
        context=bandit_feedback_continuous_train["context"],
        action=bandit_feedback_continuous_train["action"],
        reward=bandit_feedback_continuous_train["reward"],
        pscore=bandit_feedback_continuous_train["pscore"]
        )
    with open(model_partial_path+'ipw_lr_c.pickle', 'wb') as f:
        pickle.dump(ipw_lr, f)

    dataset_continuous = SyntheticBanditDataset(
        n_actions=arguments.n_actions,
        dim_context=arguments.dim_context,
        beta=2.0, # inverse temperature parameter to control the optimality and entropy of the behavior policy
        reward_type="continuous", # "binary" or "continuous"
        reward_function=reward_function_dict[arguments.reward_function],
        random_state=arguments.random_state
        )
    bandit_feedback_continuous_train = dataset_continuous.obtain_batch_bandit_feedback(n_rounds=10000)
    ipw_rf = IPWLearner(
        n_actions=dataset_continuous.n_actions,
        base_classifier=RandomForestClassifier(n_jobs=-1, random_state=arguments.random_state)
        )
    ipw_rf.fit(
        context=bandit_feedback_continuous_train["context"],
        action=bandit_feedback_continuous_train["action"],
        reward=bandit_feedback_continuous_train["reward"],
        pscore=bandit_feedback_continuous_train["pscore"]
        )
    with open(model_partial_path+'ipw_rf_c.pickle', 'wb') as f:
        pickle.dump(ipw_rf, f)

    dataset_continuous = SyntheticBanditDataset(
        n_actions=arguments.n_actions,
        dim_context=arguments.dim_context,
        beta=10.0, # inverse temperature parameter to control the optimality and entropy of the behavior policy
        reward_type="continuous", # "binary" or "continuous"
        reward_function=reward_function_dict[arguments.reward_function],
        random_state=arguments.random_state
        )
    bandit_feedback_continuous_train = dataset_continuous.obtain_batch_bandit_feedback(n_rounds=10000)
    qlr_rr = QLearner(
        n_actions=dataset_continuous.n_actions, 
        base_model=Ridge(random_state=arguments.random_state)
        )
    qlr_rr.fit(
        context=bandit_feedback_continuous_train["context"],
        action=bandit_feedback_continuous_train["action"],
        reward=bandit_feedback_continuous_train["reward"]
        )
    with open(model_partial_path+'qlr_rr_c.pickle', 'wb') as f:
        pickle.dump(qlr_rr, f)

    dataset_continuous = SyntheticBanditDataset(
        n_actions=arguments.n_actions,
        dim_context=arguments.dim_context,
        beta=-2.0, # inverse temperature parameter to control the optimality and entropy of the behavior policy
        reward_type="continuous", # "binary" or "continuous"
        reward_function=reward_function_dict[arguments.reward_function],
        random_state=arguments.random_state
        )
    bandit_feedback_continuous_train = dataset_continuous.obtain_batch_bandit_feedback(n_rounds=10000)
    qlr_rf = QLearner(
        n_actions=dataset_continuous.n_actions, 
        base_model=RandomForestRegressor(n_jobs=-1, random_state=arguments.random_state)
        )
    qlr_rf.fit(
        context=bandit_feedback_continuous_train["context"],
        action=bandit_feedback_continuous_train["action"],
        reward=bandit_feedback_continuous_train["reward"]
        )
    with open(model_partial_path+'qlr_rf_c.pickle', 'wb') as f:
        pickle.dump(qlr_rf, f)



