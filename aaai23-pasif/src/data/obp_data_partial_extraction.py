# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.

import pandas as pd
import numpy as np


def extract_partial_obp_data(batch_bandit_feedback, index):
    """create new batch bandit feedback based on specified index

    Args:
        batch_bandit_feedback (dict): batch bandit feedback which can be used in OBP
        index (array-like): Index array of samples to be extracted

    Returns:
        dict: extracted batch bandit feedback
    """

    # create extracted bandit data based on input index
    extracted_batch_bandit_feedback = {}

    for key_name in batch_bandit_feedback.keys():
        if key_name == 'n_rounds':
            extracted_batch_bandit_feedback[key_name] = len(index)
        elif key_name == 'n_actions' or key_name == 'action_context':
            extracted_batch_bandit_feedback[key_name] = batch_bandit_feedback[key_name]
        elif batch_bandit_feedback[key_name] is None:
            extracted_batch_bandit_feedback[key_name] = None
        else:
            extracted_batch_bandit_feedback[key_name] = batch_bandit_feedback[key_name][index].copy()

    return extracted_batch_bandit_feedback

