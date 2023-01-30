import pandas as pd
import numpy as np
import random

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


import os
import sys
if os.path.dirname(__file__) == '':
    sys.path.append('../../src')
else:
    sys.path.append(os.path.dirname(__file__) + '/../../src')
from data.obp_data_partial_extraction import extract_partial_obp_data


class ObpDataKfoldSplit:
    """split batch bandit feedback data into k-fold split (mainly for cross-validation)
    """

    def __init__(self, batch_bandit_feedback):
        """set batch bandit feedback

        Args:
            batch_bandit_feedback (dict): batch bandit feedback which can be used in OBP
        """

        self.batch_bandit_feedback = batch_bandit_feedback


    def set_params(self, n_repeats=1, n_splits=5, stratify=None, random_state=None):
        """set params for k-fold split

        Args:
            n_repeats (int, optional): Defaults to 1. Must be at least 1. Number of repetitions of k-fold. If n_repeats=2 and n_splits=5, we can get split data for 10 times.
            n_splits (int, optional): Defaults to 5. Number of folds (=k). Must be at least 2.
            stratify (array-like, optional): Defaults to None. If not None, data is split in a stratified fashion, using this as the class labels.
            random_state (int, optional): Defaults to None.
        """

        assert n_repeats >= 1, 'n_repeats must be at least 1'
        assert n_splits >= 2, 'n_split must be at least 2'

        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.stratify = stratify
        self.random_state = random_state

        # dict to contain index information. key:tuple (repeat, fold), value:array of index.
        self.train_index_dict = {}
        self.test_index_dict = {}

        updated_random_state = self.random_state

        for repeat in range(self.n_repeats):

            # update random_state
            if self.random_state is None:
                updated_random_state = random.randint(1, 1000000)
            else:
                updated_random_state += 1

            fold = 0
            
            if self.stratify is None:
                kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=updated_random_state)
                temp_y = np.zeros_like(self.batch_bandit_feedback['reward']) # meaningless array for kf
                for train_index, test_index in kf.split(self.batch_bandit_feedback['context'], temp_y):
                    self.train_index_dict[(repeat, fold)] = train_index
                    self.test_index_dict[(repeat, fold)] = test_index
                    fold += 1

            else:
                kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=updated_random_state)
                for train_index, test_index in kf.split(self.batch_bandit_feedback['context'], self.stratify):
                    self.train_index_dict[(repeat, fold)] = train_index
                    self.test_index_dict[(repeat, fold)] = test_index
                    fold += 1


    def get_train_test_index(self, repeat, fold):
        """get train/test index for specified repeat/fold number.

        Args:
            repeat (int): index of repeat. must be >=0 and <self.n_repeats.
            fold (int): index of fold. must be >=0 and <self.n_splits.

        Returns:
            array, array: train and test index
        """

        assert (repeat>=0) and (repeat<self.n_repeats), 'repeat must be >=0 and <self.n_repeats.'
        assert (fold>=0) and (fold<self.n_splits), 'fold must be >=0 and <self.n_splits.'

        # index of train/test
        indices_train = self.train_index_dict[(repeat, fold)]
        indices_test = self.test_index_dict[(repeat, fold)]

        return indices_train, indices_test


    def get_train_test_data(self, repeat, fold):
        """get train/test split bandit data for specified repeat/fold number.

        Args:
            repeat (int): index of repeat. must be >=0 and <self.n_repeats.
            fold (int): index of fold. must be >=0 and <self.n_splits.

        Returns:
            dict, dict: train and test batch bandit feedback which can be used in OBP
        """

        assert (repeat>=0) and (repeat<self.n_repeats), 'repeat must be >=0 and <self.n_repeats.'
        assert (fold>=0) and (fold<self.n_splits), 'fold must be >=0 and <self.n_splits.'

        # index of train/test
        indices_train, indices_test = self.get_train_test_index(repeat=repeat, fold=fold)

        # create train/test bandit data based on train/test indices
        self.train_batch_bandit_feedback = extract_partial_obp_data(batch_bandit_feedback=self.batch_bandit_feedback, index=indices_train)
        self.test_batch_bandit_feedback = extract_partial_obp_data(batch_bandit_feedback=self.batch_bandit_feedback, index=indices_test)

        return self.train_batch_bandit_feedback, self.test_batch_bandit_feedback



