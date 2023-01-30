# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.

import numpy as np
import pandas as pd
import random
import copy
from scipy import stats
from matplotlib import pyplot as plt
import time

import obp
from obp.dataset import SyntheticBanditDataset
from obp.ope import OffPolicyEvaluation, RegressionModel

import os
import sys
if os.path.dirname(__file__) == '':
    sys.path.append('../../src')
else:
    sys.path.append(os.path.dirname(__file__) + '/../../src')
from data.bootstrap_batch_feedback import sample_bootstrap_batch_bandit_feedback
from data.modify_batch_feedback import merge_batch_feedback, section_sampling_from_batch_feedback
from data.counterfactual_pi_b import get_counterfactual_action_distribution
from data.obp_data_train_test_split import ObpDataTrainTestSplit
from data.obp_data_kfold_split import ObpDataKfoldSplit
from estimator_selection.conventional_estimator_selection import ConventionalEstimatorSelection


class ConventionalPolicySelection:
    """ conventional off-policy policy selection method
    """

    def __init__(self, ope_estimators, q_models, estimator_selection_metrics='mse', data_type='synthetic', random_state=None):
        """set basic settings

        Args:
            ope_estimators (list): list of candidate estimators
            q_models (list): list of reward estimators used in model-depending estimators
            estimator_selection_metrics (str, optional): Defaults to 'mse'. Must be 'mse' or 'mean relative-ee'
            data_type (str, optional): Defaults to 'synthetic'. Must be 'synthetic' or 'real'
            random_state (int, optional): Defaults to None.
        """

        assert estimator_selection_metrics=='mse' or estimator_selection_metrics=='mean relative-ee', 'estimator_selection_metrics must be mse or mean relative-ee'
        assert data_type=='synthetic' or data_type=='real', 'data_type must be synthetic or real'

        self.ope_estimators = ope_estimators
        self.q_models = q_models
        self.estimator_selection_metrics = estimator_selection_metrics
        self.data_type = data_type
        if random_state is None:
            self.random_state = random.randint(1,10000000)
        else:
            self.random_state = random_state

    def set_real_data(
        self, 
        batch_bandit_feedback_1, 
        action_dist_1_by_2, 
        batch_bandit_feedback_2, 
        action_dist_2_by_1, 
        action_dist_1_by_pi_e, 
        action_dist_2_by_pi_e, 
        evaluation_data='partial_random'
        ):
        """set real-world data (logged bandit feedback)

        Args:
            batch_bandit_feedback_1 (dict): batch bandit feedback 1
            action_dist_1_by_2 (np.array): action distribution for data 1 by policy 2
            batch_bandit_feedback_2 (dict): batch bandit feedback 2
            action_dist_2_by_1 (np.array): action distribution for data 2 by policy 1
            action_dist_1_by_pi_e (dict): dictionary of evaluation policies. keys: name of policy (str). values: array of pi_e for data 1.
            action_dist_2_by_pi_e (dict): dictionary of evaluation policies. keys: name of policy (str). values: array of pi_e for data 2.
            evaluation_data (str, optional): Defaults to 'random'. Must be '1' or '2' or 'random' or 'partial_random'. 
                                             Which data (behavior policy) to consider as evaluation policy.
                                             'partial_random' means that we use fixed data as evalu policy in bootstrap, but no fixed in outer loop in estimator selection.
        """
        assert action_dist_1_by_pi_e.keys() == action_dist_2_by_pi_e.keys(), 'action_dist_1_by_pi_e and action_dist_2_by_pi_e must have same keys'

        self.batch_bandit_feedback_1 = batch_bandit_feedback_1
        self.action_dist_1_by_2 = action_dist_1_by_2
        self.batch_bandit_feedback_2 = batch_bandit_feedback_2
        self.action_dist_2_by_1 = action_dist_2_by_1
        self.action_dist_1_by_pi_e = action_dist_1_by_pi_e
        self.action_dist_2_by_pi_e = action_dist_2_by_pi_e
        self.evaluation_data = evaluation_data


    def set_synthetic_data(self, dataset_1, n_rounds_1, dataset_2, n_rounds_2, pi_e, evaluation_data='partial_random'):
        """set synthetic data

        Args:
            dataset_1 (obp.dataset.SyntheticBanditDataset): synthetic data generator 1
            n_rounds_1 (int): sample size of data 1
            dataset_2 (obp.dataset.SyntheticBanditDataset): synthetic data generator 2
            n_rounds_2 (int): sample size of data 2
            pi_e (dict): dictionary of evaluation policies. keys: name of policy (str). values: tuple including info of evaluation policy.
                          ex. {'beta_1.0':('beta', 1.0), 'function_1':('function', pi(a|x), tau)} 
                          * ('beta', 1.0) (using beta to specify evaluation policy)
                          * ('function', pi(a|x), tau) (Give any function as evaluation policy. To get action_diost, we use predict_proba(tau=tau))
            evaluation_data (str, optional): Defaults to 'random'. Must be '1' or '2' or 'random' or 'partial_random'. 
                                             Which data (behavior policy) to consider as evaluation policy.
                                             'partial_random' means that we use fixed data as evalu policy in bootstrap, but no fixed in outer loop in estimator selection.
        """
        assert dataset_1.random_state == dataset_2.random_state, 'recommend dataset_1.random_state == dataset_2.random_state'
        self.dataset_1 = dataset_1
        self.n_rounds_1 = n_rounds_1
        self.dataset_2 = dataset_2
        self.n_rounds_2 = n_rounds_2
        self.pi_e = pi_e
        self.evaluation_data = evaluation_data

        self.original_data_seed_1 = self.dataset_1.random_state
        self.original_data_seed_2 = self.dataset_2.random_state



    def _evaluate_policies_single_outer_loop(
        self, 
        batch_bandit_feedback_1_train, 
        action_dist_1_by_2_train, 
        batch_bandit_feedback_2_train, 
        action_dist_2_by_1_train, 
        batch_bandit_feedback_1_test, 
        action_dist_1_by_2_test, 
        batch_bandit_feedback_2_test, 
        action_dist_2_by_1_test, 
        action_dist_1_by_pi_e_test, 
        action_dist_2_by_pi_e_test, 
        evaluation_data='random', 
        n_bootstrap=100
        ):
        """For given batch data and evaluation policies, we estimate estimator performance with bootstrap and select best policy

        Args:
            batch_bandit_feedback_1_train (dict): batch bandit feedback 1 for train (estimator selection)
            action_dist_1_by_2_train (np.array): action distribution for data 1 by policy 2 (train data for estimator selection)
            batch_bandit_feedback_2_train (dict): batch bandit feedback 2 for train (estimator selection)
            action_dist_2_by_1_train (np.array): action distribution for data 2 by policy 1 (train data for estimator selection)
            batch_bandit_feedback_1_test (dict): batch bandit feedback 1 for test (ope and policy selection)
            action_dist_1_by_2_test (np.array): action distribution for data 1 by policy 2 (test data for policy selection)
            batch_bandit_feedback_2_test (dict): batch bandit feedback 2 for test (ope and policy selection)
            action_dist_2_by_1_test (np.array): action distribution for data 2 by policy 1 (test data for policy selection)
            action_dist_1_by_pi_e_test (dict): dictionary of evaluation policies. keys: name of policy (str). values: array of pi_e for test data 1.
            action_dist_2_by_pi_e_test (dict): dictionary of evaluation policies. keys: name of policy (str). values: array of pi_e for test data 2.
            evaluation_data (str, optional): Defaults to 'random'. Must be '1' or '2' or 'random'. Which data (behavior policy) to consider as evaluation policy in estimator selection.
            n_bootstrap (int, optional): Defaults to 100. The number of bootstrap sampling in estimator selection. If None, we use original data only once.
        Returns:
            pd.DataFrame, dict: dataframe with columns=[policy_name, estimator_name, estimated_policy_value, rank], and estimator selection result for each evaluation policy
        """

        policy_performance = pd.DataFrame(columns=['policy_name', 'estimator_name', 'estimated_policy_value'])
        estimator_selection_result = {} #key:policy_name, value:df (estimator_name, mean mse, rank, estimated_policy_value_for_test_data)

        # estimator selection
        conventional_estimator_selection = ConventionalEstimatorSelection(
            ope_estimators=self.ope_estimators, 
            q_models=self.q_models, 
            metrics=self.estimator_selection_metrics, 
            data_type='real', 
            random_state=self.random_state
            )
        conventional_estimator_selection.set_real_data(
            batch_bandit_feedback_1=batch_bandit_feedback_1_train, 
            action_dist_1_by_2=action_dist_1_by_2_train, 
            batch_bandit_feedback_2=batch_bandit_feedback_2_train, 
            action_dist_2_by_1=action_dist_2_by_1_train, 
            evaluation_data=evaluation_data
            )
        conventional_estimator_selection.evaluate_estimators(
            n_inner_bootstrap=n_bootstrap, 
            n_outer_bootstrap=None, 
            outer_repeat_type='bootstrap', 
            ground_truth_method='on_policy'
            )
        summarized_estimator_selection_result = conventional_estimator_selection.get_summarized_results()[['estimator_name', 'mean '+self.estimator_selection_metrics, 'rank']]
        best_estimator, best_q_model = conventional_estimator_selection.get_best_estimator()
        if 'IPW' in best_estimator.estimator_name:
            best_estimator_name = best_estimator.estimator_name
        else:
            best_estimator_name = best_estimator.estimator_name + '_qmodel_' + str(best_q_model)
            
        # estimate policy value
        ## merge data
        merged_test_data = merge_batch_feedback(
            batch_bandit_feedback_1=batch_bandit_feedback_1_test, 
            action_dist_1_by_2=action_dist_1_by_2_test, 
            batch_bandit_feedback_2=batch_bandit_feedback_2_test, 
            action_dist_2_by_1=action_dist_2_by_1_test, 
            )
        
        for policy_name in action_dist_1_by_pi_e_test.keys():
            merged_action_dist_by_pi_e = np.concatenate([action_dist_1_by_pi_e_test[policy_name], action_dist_2_by_pi_e_test[policy_name]])
            estimated_policy_value = {}

            for i, q_model in enumerate(self.q_models):
                renamed_ope_estimators = []
                
                for ope_estimator in copy.deepcopy(self.ope_estimators):
                    if 'IPW' in ope_estimator.estimator_name:
                        if i == 0:
                            renamed_ope_estimators.append(ope_estimator)
                        else:
                            pass
                    else:
                        new_estimator_name = ope_estimator.estimator_name + '_qmodel_' + str(q_model)
                        ope_estimator.estimator_name = new_estimator_name
                        renamed_ope_estimators.append(ope_estimator)
                
                ## conduct ope for each evaluation policy
                regression_model = RegressionModel(
                    n_actions=merged_test_data['n_actions'],
                    action_context=merged_test_data['action_context'],
                    base_model=q_model(random_state=self.random_state)
                    )
                estimated_rewards_by_reg_model = regression_model.fit_predict(
                    context=merged_test_data["context"],
                    action=merged_test_data["action"],
                    reward=merged_test_data["reward"],
                    n_folds=3, # use 3-fold cross-fitting
                    random_state=self.random_state
                    )

                ope = OffPolicyEvaluation(
                    bandit_feedback=merged_test_data,
                    ope_estimators=renamed_ope_estimators
                    )
                estimated_policy_value.update( ope.estimate_policy_values(
                    action_dist=merged_action_dist_by_pi_e,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
                    ) )

            policy_performance = policy_performance.append({
                'policy_name':policy_name, 
                'estimator_name':best_estimator_name, 
                'estimated_policy_value':estimated_policy_value[best_estimator_name]
                }, ignore_index=True)

            estimated_policy_value = pd.DataFrame({'estimator_name':estimated_policy_value.keys(), 'estimated_policy_value_for_test_data':estimated_policy_value.values()})
            estimator_selection_result[policy_name] = pd.merge(copy.deepcopy(summarized_estimator_selection_result), estimated_policy_value, how='left', on='estimator_name')

        # add rank
        policy_rank = policy_performance.rank(method='min', ascending=False)[['estimated_policy_value']].rename(columns={'estimated_policy_value':'rank'})
        policy_performance = pd.merge(policy_performance, policy_rank, left_index=True, right_index=True)
            
        return policy_performance, estimator_selection_result


    def evaluate_policies(
        self, 
        n_inner_bootstrap, 
        n_outer_loop=10, 
        outer_loop_type='uniform_sampling', 
        test_ratio=0.5
        ):
        """for set data, we evaluate policies (with bootstrappng estimator selection) for several times

        Args:
            n_inner_bootstrap (int): The number of bootstrap sampling in ope estimator selection. If None, we use original data only once.
            n_outer_loop (int, optional): Defaults to 10. The number of policy selection. 
            outer_loop_type (str, optional): Defaults to 'uniform_sampling'. Must be 'uniform_sampling' or 'cross_validation' or 'generation'.
                                             If we use synthetic data, we can use 'generation' meaning generating data for each outer loop.
            test_ratio (float, optional): Defaults to 0.5. Ratio of test set. This is valid when outer_loop_type is 'uniform_sampling' or 'generation'. 
                                          If None, we use whole data for both estimator selection and policy selection (valid when outer_loop_type='generation').
        """
        assert outer_loop_type=='uniform_sampling' or outer_loop_type=='cross_validation' or outer_loop_type=='generation', \
            'outer_loop_type must be uniform_sampling/cross_validation/generation'
        if outer_loop_type=='uniform_sampling' or  outer_loop_type=='generation':
            assert n_outer_loop >= 1, 'if outer_loop_type is uniform_sampling/generation, n_outer_loop >= 1'
        elif outer_loop_type=='cross_validation':
            assert n_outer_loop >= 2, 'if outer_loop_type is cross_validation, n_outer_loop >= 2'

        if self.data_type == 'synthetic':
            self.dataset_2.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_2) # to get different features between data 1 and 2

            self.action_dist_1_by_pi_e = {}
            self.action_dist_2_by_pi_e = {}

            for policy_name, pi_e in self.pi_e.items():
                if pi_e[0] == 'beta':
                    self.action_dist_1_by_pi_e[policy_name] = get_counterfactual_action_distribution(dataset=self.dataset_1, cf_beta=pi_e[1], n_rounds=self.n_rounds_1)
                    self.action_dist_2_by_pi_e[policy_name] = get_counterfactual_action_distribution(dataset=self.dataset_2, cf_beta=pi_e[1], n_rounds=self.n_rounds_2)
            
            self.action_dist_1_by_2 = get_counterfactual_action_distribution(dataset=self.dataset_1, cf_beta=self.dataset_2.beta, n_rounds=self.n_rounds_1)
            self.batch_bandit_feedback_1 = self.dataset_1.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_1)
            self.action_dist_2_by_1 = get_counterfactual_action_distribution(dataset=self.dataset_2, cf_beta=self.dataset_1.beta, n_rounds=self.n_rounds_2)
            self.batch_bandit_feedback_2 = self.dataset_2.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_2)

            for policy_name, pi_e in self.pi_e.items():
                if pi_e[0] == 'function':
                    action_dist_1_by_pi_e = pi_e[1].predict_proba(self.batch_bandit_feedback_1['context'], tau=pi_e[2])
                    self.action_dist_1_by_pi_e[policy_name] = action_dist_1_by_pi_e.reshape((action_dist_1_by_pi_e.shape[0], action_dist_1_by_pi_e.shape[1], 1))
                    action_dist_2_by_pi_e = pi_e[1].predict_proba(self.batch_bandit_feedback_2['context'], tau=pi_e[2])
                    self.action_dist_2_by_pi_e[policy_name] = action_dist_2_by_pi_e.reshape((action_dist_2_by_pi_e.shape[0], action_dist_2_by_pi_e.shape[1], 1))

        if outer_loop_type == 'cross_validation':
            CrossValidation1 = ObpDataKfoldSplit(self.batch_bandit_feedback_1)
            CrossValidation1.set_params(n_repeats=1, n_splits=n_outer_loop, stratify=None, random_state=self.random_state)
            CrossValidation2 = ObpDataKfoldSplit(self.batch_bandit_feedback_2)
            CrossValidation2.set_params(n_repeats=1, n_splits=n_outer_loop, stratify=None, random_state=self.random_state)

        for i_outer in range(n_outer_loop):
            print('conventional_policy_selection outer loop ', i_outer, time.gmtime())
            if outer_loop_type == 'uniform_sampling':
                TrainTestSplit = ObpDataTrainTestSplit(self.batch_bandit_feedback_1)
                TrainTestSplit.set_params(test_size=test_ratio, random_state=self.random_state+i_outer, stratify=None)
                indices_train, indices_test = TrainTestSplit.get_train_test_index()
                batch_bandit_feedback_1_train, batch_bandit_feedback_1_test = TrainTestSplit.get_train_test_data()
                action_dist_1_by_2_train = self.action_dist_1_by_2[indices_train]
                action_dist_1_by_2_test = self.action_dist_1_by_2[indices_test]
                action_dist_1_by_pi_e_test = {}
                for policy_name, action_dist in self.action_dist_1_by_pi_e.items():
                    action_dist_1_by_pi_e_test[policy_name] = action_dist[indices_test]

                TrainTestSplit = ObpDataTrainTestSplit(self.batch_bandit_feedback_2)
                TrainTestSplit.set_params(test_size=test_ratio, random_state=self.random_state+i_outer, stratify=None)
                indices_train, indices_test = TrainTestSplit.get_train_test_index()
                batch_bandit_feedback_2_train, batch_bandit_feedback_2_test = TrainTestSplit.get_train_test_data()
                action_dist_2_by_1_train = self.action_dist_2_by_1[indices_train]
                action_dist_2_by_1_test = self.action_dist_2_by_1[indices_test]
                action_dist_2_by_pi_e_test = {}
                for policy_name, action_dist in self.action_dist_2_by_pi_e.items():
                    action_dist_2_by_pi_e_test[policy_name] = action_dist[indices_test]

            elif outer_loop_type == 'cross_validation':
                indices_train, indices_test = CrossValidation1.get_train_test_index(repeat=0, fold=i_outer)
                batch_bandit_feedback_1_train, batch_bandit_feedback_1_test = CrossValidation1.get_train_test_data(repeat=0, fold=i_outer)
                action_dist_1_by_2_train = self.action_dist_1_by_2[indices_train]
                action_dist_1_by_2_test = self.action_dist_1_by_2[indices_test]
                action_dist_1_by_pi_e_test = {}
                for policy_name, action_dist in self.action_dist_1_by_pi_e.items():
                    action_dist_1_by_pi_e_test[policy_name] = action_dist[indices_test]

                indices_train, indices_test = CrossValidation2.get_train_test_index(repeat=0, fold=i_outer)
                batch_bandit_feedback_2_train, batch_bandit_feedback_2_test = CrossValidation2.get_train_test_data(repeat=0, fold=i_outer)
                action_dist_2_by_1_train = self.action_dist_2_by_1[indices_train]
                action_dist_2_by_1_test = self.action_dist_2_by_1[indices_test]
                action_dist_2_by_pi_e_test = {}
                for policy_name, action_dist in self.action_dist_2_by_pi_e.items():
                    action_dist_2_by_pi_e_test[policy_name] = action_dist[indices_test]

            elif outer_loop_type == 'generation':

                self.action_dist_1_by_pi_e = {}
                self.action_dist_2_by_pi_e = {}

                for policy_name, pi_e in self.pi_e.items():
                    if pi_e[0] == 'beta':
                        self.action_dist_1_by_pi_e[policy_name] = get_counterfactual_action_distribution(dataset=self.dataset_1, cf_beta=pi_e[1], n_rounds=self.n_rounds_1)
                        self.action_dist_2_by_pi_e[policy_name] = get_counterfactual_action_distribution(dataset=self.dataset_2, cf_beta=pi_e[1], n_rounds=self.n_rounds_2)
                
                self.action_dist_1_by_2 = get_counterfactual_action_distribution(dataset=self.dataset_1, cf_beta=self.dataset_2.beta, n_rounds=self.n_rounds_1)
                self.batch_bandit_feedback_1 = self.dataset_1.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_1)
                self.action_dist_2_by_1 = get_counterfactual_action_distribution(dataset=self.dataset_2, cf_beta=self.dataset_1.beta, n_rounds=self.n_rounds_2)
                self.batch_bandit_feedback_2 = self.dataset_2.obtain_batch_bandit_feedback(n_rounds=self.n_rounds_2)

                for policy_name, pi_e in self.pi_e.items():
                    if pi_e[0] == 'function':
                        action_dist_1_by_pi_e = pi_e[1].predict_proba(self.batch_bandit_feedback_1['context'], tau=pi_e[2])
                        self.action_dist_1_by_pi_e[policy_name] = action_dist_1_by_pi_e.reshape((action_dist_1_by_pi_e.shape[0], action_dist_1_by_pi_e.shape[1], 1))
                        action_dist_2_by_pi_e = pi_e[1].predict_proba(self.batch_bandit_feedback_2['context'], tau=pi_e[2])
                        self.action_dist_2_by_pi_e[policy_name] = action_dist_2_by_pi_e.reshape((action_dist_2_by_pi_e.shape[0], action_dist_2_by_pi_e.shape[1], 1))
                if test_ratio is None:
                    pass
                else:
                    TrainTestSplit = ObpDataTrainTestSplit(self.batch_bandit_feedback_1)
                    TrainTestSplit.set_params(test_size=test_ratio, random_state=self.random_state+i_outer, stratify=None)
                    indices_train, indices_test = TrainTestSplit.get_train_test_index()
                    batch_bandit_feedback_1_train, batch_bandit_feedback_1_test = TrainTestSplit.get_train_test_data()
                    action_dist_1_by_2_train = self.action_dist_1_by_2[indices_train]
                    action_dist_1_by_2_test = self.action_dist_1_by_2[indices_test]
                    action_dist_1_by_pi_e_test = {}
                    for policy_name, action_dist in self.action_dist_1_by_pi_e.items():
                        action_dist_1_by_pi_e_test[policy_name] = action_dist[indices_test]

                    TrainTestSplit = ObpDataTrainTestSplit(self.batch_bandit_feedback_2)
                    TrainTestSplit.set_params(test_size=test_ratio, random_state=self.random_state+i_outer, stratify=None)
                    indices_train, indices_test = TrainTestSplit.get_train_test_index()
                    batch_bandit_feedback_2_train, batch_bandit_feedback_2_test = TrainTestSplit.get_train_test_data()
                    action_dist_2_by_1_train = self.action_dist_2_by_1[indices_train]
                    action_dist_2_by_1_test = self.action_dist_2_by_1[indices_test]
                    action_dist_2_by_pi_e_test = {}
                    for policy_name, action_dist in self.action_dist_2_by_pi_e.items():
                        action_dist_2_by_pi_e_test[policy_name] = action_dist[indices_test]

            if self.evaluation_data == 'partial_random':
                random.seed(self.random_state+i_outer)
                evaluation_data = random.choice(['1', '2'])
            else:
                evaluation_data = self.evaluation_data

            if test_ratio is None:
                policy_performance, estimator_selection_result = self._evaluate_policies_single_outer_loop(
                    batch_bandit_feedback_1_train=self.batch_bandit_feedback_1, 
                    action_dist_1_by_2_train=self.action_dist_1_by_2, 
                    batch_bandit_feedback_2_train=self.batch_bandit_feedback_2, 
                    action_dist_2_by_1_train=self.action_dist_2_by_1, 
                    batch_bandit_feedback_1_test=self.batch_bandit_feedback_1, 
                    action_dist_1_by_2_test=self.action_dist_1_by_2, 
                    batch_bandit_feedback_2_test=self.batch_bandit_feedback_2, 
                    action_dist_2_by_1_test=self.action_dist_2_by_1, 
                    action_dist_1_by_pi_e_test=self.action_dist_1_by_pi_e, 
                    action_dist_2_by_pi_e_test=self.action_dist_2_by_pi_e, 
                    evaluation_data=evaluation_data, 
                    n_bootstrap=n_inner_bootstrap
                    )
            else:
                policy_performance, estimator_selection_result = self._evaluate_policies_single_outer_loop(
                    batch_bandit_feedback_1_train=batch_bandit_feedback_1_train, 
                    action_dist_1_by_2_train=action_dist_1_by_2_train, 
                    batch_bandit_feedback_2_train=batch_bandit_feedback_2_train, 
                    action_dist_2_by_1_train=action_dist_2_by_1_train, 
                    batch_bandit_feedback_1_test=batch_bandit_feedback_1_test, 
                    action_dist_1_by_2_test=action_dist_1_by_2_test, 
                    batch_bandit_feedback_2_test=batch_bandit_feedback_2_test, 
                    action_dist_2_by_1_test=action_dist_2_by_1_test, 
                    action_dist_1_by_pi_e_test=action_dist_1_by_pi_e_test, 
                    action_dist_2_by_pi_e_test=action_dist_2_by_pi_e_test, 
                    evaluation_data=evaluation_data, 
                    n_bootstrap=n_inner_bootstrap
                    )

            policy_performance['outer_iteration'] = i_outer
            for p_name, e_selection_df in estimator_selection_result.items():
                e_selection_df['outer_iteration'] = i_outer
                estimator_selection_result[p_name] = e_selection_df

            if i_outer == 0:
                self.all_result = policy_performance
                self.all_estimator_selection_result = estimator_selection_result
            else:
                self.all_result = pd.concat([self.all_result, policy_performance])
                for p_name, e_selection_df in estimator_selection_result.items():
                    self.all_estimator_selection_result[p_name] = pd.concat([self.all_estimator_selection_result[p_name], e_selection_df])


        self.all_result = self.all_result[['outer_iteration', 'policy_name', 'estimator_name', 'estimated_policy_value', 'rank']]
        self.summarized_result = pd.DataFrame(columns=['policy_name','mean estimated_policy_value', 'stdev', '95%CI(upper)', '95%CI(lower)'])
        for policy_name in self.all_result['policy_name'].unique():
            summarized_result = [policy_name]
            summarized_result.append(self.all_result['estimated_policy_value'][self.all_result['policy_name']==policy_name].mean())
            if self.all_result['outer_iteration'].max() > 0:
                summarized_result.append(self.all_result['estimated_policy_value'][self.all_result['policy_name']==policy_name].std())
                t_dist = stats.t(loc=summarized_result[1],
                                scale=stats.sem(self.all_result['estimated_policy_value'][self.all_result['policy_name']==policy_name]),
                                df=len(self.all_result['estimated_policy_value'][self.all_result['policy_name']==policy_name])-1)
                bottom, up = t_dist.interval(alpha=0.95)
                summarized_result.append(up)
                summarized_result.append(bottom)
            else:
                summarized_result.append(None)
                summarized_result.append(None)
                summarized_result.append(None)
            self.summarized_result = self.summarized_result.append({
                'policy_name':summarized_result[0],
                'mean estimated_policy_value':summarized_result[1], 
                'stdev':summarized_result[2], 
                '95%CI(upper)':summarized_result[3], 
                '95%CI(lower)':summarized_result[4]
                }, ignore_index=True)

        policy_rank = self.summarized_result.rank(method='min', ascending=False)[['mean estimated_policy_value']].rename(columns={'mean estimated_policy_value':'rank'})
        self.summarized_result = pd.merge(self.summarized_result, policy_rank, left_index=True, right_index=True)
            

    def get_all_estimator_selection_results(self):
        """get all estimator selection results (all outer lopp results for all evaluation policy)

        Returns:
            dict: results for all outer loop. key: policy name, value: df.
        """
        return self.all_estimator_selection_result

    def get_all_results(self):
        """get all policy selection results (all outer lopp results)

        Returns:
            pd.DataFrame: results for all outer loop
        """
        return self.all_result

    def get_summarized_results(self):
        """get summarized policy selection results

        Returns:
            pd.DataFrame: results (mean results for outer loops)
        """
        return self.summarized_result

    def get_best_estimator(self):
        """get best policy

        Returns:
            str: name of best policy
        """
        best_policy_name = self.summarized_result['policy_name'][self.summarized_result['rank']==1].values[0]
        return best_policy_name


    def visualize_results(self, show=False, save_path=None):
        """show bar-graph of policy value (with 95% CI)

        Args:
            show (bool, optional): Defaults to False. If True, show figure.
            save_path (str, optional): Defaults to None. If None, we do not save figure. If not None, we save figure in save_path.
        """

        # error range of 95% CI
        err_range = []
        err_range.append(self.summarized_result['mean estimated_policy_value'] - self.summarized_result['95%CI(lower)'])
        err_range.append(self.summarized_result['95%CI(upper)'] - self.summarized_result['mean estimated_policy_value'])
        
        # plot the result
        plt.figure()
        plt.bar(
            self.summarized_result['policy_name'], 
            self.summarized_result['mean estimated_policy_value'], 
            yerr=err_range
            )
        plt.xlabel('policy')
        plt.ylabel('estimated policy value')
        if show:
            plt.show()
        if not(save_path is None):
            plt.savefig(save_path)


