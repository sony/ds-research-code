# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.

import copy

def get_counterfactual_action_distribution(dataset, cf_beta, n_rounds):
    """get action distribution for counterfactual beta.
    Note that we need to use this function before we get factual batch data by dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    Args:
        dataset (obp.dataset.SyntheticBanditDataset): original synthetic data generator
        cf_beta (float): counterfactual beta
        n_rounds (int): sample size

    Returns:
        np.array: action distribution for counterfactual beta
    """
    cf_dataset = copy.deepcopy(dataset)
    setattr(cf_dataset, 'beta', cf_beta)
    return cf_dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)['pi_b']

