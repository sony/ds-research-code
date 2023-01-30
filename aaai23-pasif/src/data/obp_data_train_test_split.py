import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


import os
import sys
if os.path.dirname(__file__) == '':
    sys.path.append('../../src')
else:
    sys.path.append(os.path.dirname(__file__) + '/../../src')
from data.obp_data_partial_extraction import extract_partial_obp_data


class ObpDataTrainTestSplit:
    """split batch bandit feedback data into train test data
    """

    def __init__(self, batch_bandit_feedback):
        """set batch bandit feedback

        Args:
            batch_bandit_feedback (dict): batch bandit feedback which can be used in OBP
        """

        self.batch_bandit_feedback = batch_bandit_feedback


    def set_params(self, test_size=0.25, random_state=None, stratify=None):
        """set params for train/test split

        Args:
            test_size (float, optional): Defaults to 0.25. Must be >0 and <1.
            random_state (int, optional): Defaults to None.
            stratify (array-like, optional): Defaults to None. If not None, data is split in a stratified fashion, using this as the class labels.
        """

        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

        # split indec into train/test
        indices = np.array(range(self.batch_bandit_feedback['n_rounds']))
        self.indices_train, self.indices_test = train_test_split(indices, test_size=self.test_size, random_state=self.random_state, stratify=self.stratify)

    def get_train_test_index(self):
        """get train/test index

        Returns:
            array, array: train and test index
        """

        return self.indices_train, self.indices_test

    def get_train_test_data(self):
        """get train/test split bandit data

        Returns:
            dict, dict: train and test batch bandit feedback which can be used in OBP
        """
        # create train/test bandit data based on split indices
        self.train_batch_bandit_feedback = extract_partial_obp_data(batch_bandit_feedback=self.batch_bandit_feedback, index=self.indices_train)
        self.test_batch_bandit_feedback = extract_partial_obp_data(batch_bandit_feedback=self.batch_bandit_feedback, index=self.indices_test)

        return self.train_batch_bandit_feedback, self.test_batch_bandit_feedback



