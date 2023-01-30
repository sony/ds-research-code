import numpy as np
import pandas as pd


def section_sampling_from_batch_feedback(batch_bandit_feedback, section_start=0, section_end=100):
    """sampling from batch_bandit_feedback and return sampled batch_bandit_feedback

    Args:
        batch_bandit_feedback (dict): batch bandit feedback
        section_start (int, optional): Defaults to 0. index of start sample.
        section_end (int, optional): Defaults to 100. index of final sample. 

    Returns:
        dict: sampled batch bandit feedback
    """
    
    sampled_batch_bandit_feedback = {}
    
    for feedback_key in batch_bandit_feedback.keys():
        if feedback_key == 'n_rounds':
            sampled_batch_bandit_feedback[feedback_key] = section_end - section_start
        elif feedback_key == 'n_actions':
            sampled_batch_bandit_feedback[feedback_key] = batch_bandit_feedback[feedback_key]
        elif feedback_key == 'context':
            sampled_batch_bandit_feedback[feedback_key] = batch_bandit_feedback[feedback_key][section_start:section_end, :].copy()
        elif feedback_key == 'action_context':
            if batch_bandit_feedback[feedback_key] is None:
                sampled_batch_bandit_feedback[feedback_key] = None
            else:
                sampled_batch_bandit_feedback[feedback_key] = batch_bandit_feedback[feedback_key][section_start:section_end, :].copy()
        elif feedback_key == 'action':
            sampled_batch_bandit_feedback[feedback_key] = batch_bandit_feedback[feedback_key][section_start:section_end].copy()
        elif feedback_key == 'position':
            if batch_bandit_feedback[feedback_key] is None:
                sampled_batch_bandit_feedback[feedback_key] = None
            else:
                sampled_batch_bandit_feedback[feedback_key] = batch_bandit_feedback[feedback_key][section_start:section_end].copy()
        elif feedback_key == 'reward':
            sampled_batch_bandit_feedback[feedback_key] = batch_bandit_feedback[feedback_key][section_start:section_end].copy()
        elif feedback_key == 'expected_reward':
            if not(feedback_key in list(batch_bandit_feedback.keys())):
                pass
            elif batch_bandit_feedback[feedback_key] is None:
                sampled_batch_bandit_feedback[feedback_key] = None
            else:
                sampled_batch_bandit_feedback[feedback_key] = batch_bandit_feedback[feedback_key][section_start:section_end, :].copy()
        elif feedback_key == 'pi_b':
            if batch_bandit_feedback[feedback_key] is None:
                sampled_batch_bandit_feedback[feedback_key] = None
            else:
                sampled_batch_bandit_feedback[feedback_key] = batch_bandit_feedback[feedback_key][section_start:section_end, :].copy()
        elif feedback_key == 'pscore':
            sampled_batch_bandit_feedback[feedback_key] = batch_bandit_feedback[feedback_key][section_start:section_end].copy()
    
    return sampled_batch_bandit_feedback




def merge_batch_feedback(
    batch_bandit_feedback_1, 
    action_dist_1_by_2, 
    batch_bandit_feedback_2, 
    action_dist_2_by_1, 
    ):
    """merge 2 batch feedback and return merged batch

    Args:
        batch_bandit_feedback_1 (dict): batch bandit feedback 1
        action_dist_1_by_2 (np.array): action distribution for data 1 by policy 2
        batch_bandit_feedback_2 (dict): batch bandit feedback 2
        action_dist_2_by_1 (np.array): action distribution for data 2 by policy 1

    Returns:
        dict: merged batch bandit feedback
    """

    merged_batch = {}
    merged_batch['n_rounds'] = batch_bandit_feedback_1['n_rounds'] + batch_bandit_feedback_2['n_rounds']
    merged_batch['n_actions'] = batch_bandit_feedback_1['n_actions']
    merged_batch['context'] = np.concatenate([batch_bandit_feedback_1['context'], batch_bandit_feedback_2['context']]).copy()
    merged_batch['action_context'] = batch_bandit_feedback_1['action_context']
    merged_batch['action'] = np.concatenate([batch_bandit_feedback_1['action'], batch_bandit_feedback_2['action']]).copy()
    if batch_bandit_feedback_1['position'] is None:
        merged_batch['position'] = None
    else:
        merged_batch['position'] = np.concatenate([batch_bandit_feedback_1['position'], batch_bandit_feedback_2['position']]).copy()
    merged_batch['reward'] = np.concatenate([batch_bandit_feedback_1['reward'], batch_bandit_feedback_2['reward']]).copy()
    if not('expected_reward' in list(batch_bandit_feedback_1.keys())):
        pass
    else:
        merged_batch['expected_reward'] = np.concatenate([batch_bandit_feedback_1['expected_reward'], batch_bandit_feedback_2['expected_reward']]).copy()

    pi_b_1 = \
        (batch_bandit_feedback_1['n_rounds']/merged_batch['n_rounds']) * batch_bandit_feedback_1['pi_b'] + \
        (batch_bandit_feedback_2['n_rounds']/merged_batch['n_rounds']) * action_dist_1_by_2
    pi_b_2 = \
        (batch_bandit_feedback_1['n_rounds']/merged_batch['n_rounds']) * action_dist_2_by_1 + \
        (batch_bandit_feedback_2['n_rounds']/merged_batch['n_rounds']) * batch_bandit_feedback_2['pi_b']
    merged_batch['pi_b'] = np.concatenate([pi_b_1, pi_b_2])

    p_score = np.zeros(merged_batch['n_rounds'])
    for i in range(merged_batch['n_actions']):
        p_score[merged_batch['action']==i] = merged_batch['pi_b'][merged_batch['action']==i, i, 0].copy()
    merged_batch['pscore'] = p_score

    return merged_batch



