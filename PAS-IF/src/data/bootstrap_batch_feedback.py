import pandas as pd
import numpy as np
import copy

from sklearn.utils import check_random_state

def sample_bootstrap_batch_bandit_feedback(
    batch_bandit_feedback, 
    action_dist=None, 
    sample_size_ratio=1.0,
    random_state=None
    ):
    """get bootstrapped batch bandit feedback

    Args:
        batch_bandit_feedback (dict): batch bandit feedback
        action_dist (np.array): action distribution for batch_bandit_feedback by other policy
        sample_size_ratio (float, optional): Defaults to 1.0. The ratio of the sample size of the sampled data to that of the original data
        random_state (int, optional): Defaults to None.

    Returns:
        dict: bootstrapped batch bandit feedback
    """

    bootstrapped_data = copy.deepcopy(batch_bandit_feedback)
    bootstrapped_data['n_rounds'] = int(batch_bandit_feedback['n_rounds']*sample_size_ratio)

    bootstrap_idx = check_random_state(random_state).choice(np.arange(batch_bandit_feedback['n_rounds']), size=bootstrapped_data['n_rounds'], replace=True)
    for key_name in ['context', 'action', 'position', 'reward', 'expected_reward', 'pi_b', 'pscore']:
        if not(key_name in list(batch_bandit_feedback.keys())):
            pass
        elif batch_bandit_feedback[key_name] is None:
            pass
        else:
            bootstrapped_data[key_name] = batch_bandit_feedback[key_name][bootstrap_idx]
    
    bootstrapped_action_dist = None
    if not(action_dist is None):
        bootstrapped_action_dist = action_dist.copy()[bootstrap_idx]

    return bootstrapped_data, bootstrapped_action_dist