# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.

from audioop import add
import numpy as np
import pandas as pd
import random
import copy
from scipy import stats
from matplotlib import pyplot as plt

import torch.optim as optim

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
from data.counterfactual_pi_b import get_counterfactual_action_distribution
from estimator_selection.conventional_estimator_selection import ConventionalEstimatorSelection
from estimator_selection.data_split_pass import DataSplittingByPass
from estimator_selection.data_split_pasif import DataSplittingByPasif


class ProposedEstimatorSelection:
    """ new (proposed) off-policy estimator selection method
    """

    def __init__(self, ope_estimators, q_models, metrics='mse', data_type='synthetic', random_state=None):
        """set basic settings

        Args:
            ope_estimators (list): list of candidate estimators
            q_models (list): list of reward estimators used in model-depending estimators
            metrics (str, optional): Defaults to 'mse'. Must be 'mse' or 'mean relative-ee'
            data_type (str, optional): Defaults to 'synthetic'. Must be 'synthetic' or 'real'
            random_state (int, optional): Defaults to None.
        """

        assert metrics=='mse' or metrics=='mean relative-ee', 'metrics must be mse or mean relative-ee'
        assert data_type=='synthetic' or data_type=='real', 'data_type must be synthetic or real'

        self.ope_estimators = ope_estimators
        self.q_models = q_models
        self.metrics = metrics
        self.data_type = data_type
        if random_state is None:
            self.random_state = random.randint(1,10000000)
        else:
            self.random_state = random_state

    def set_real_data(
        self, 
        batch_bandit_feedback, 
        action_dist_by_pi_e
        ):
        """set real-world data (logged bandit feedback)

        Args:
            batch_bandit_feedback (dict): batch bandit feedback
            action_dist_pi_e (np.array): action distribution for batch bandit feedback by evaluation policy
        """
        self.batch_bandit_feedback = batch_bandit_feedback
        self.action_dist_by_pi_e = action_dist_by_pi_e


    def set_synthetic_data(self, dataset, n_rounds, pi_e):
        """set synthetic data

        Args:
            dataset (obp.dataset.SyntheticBanditDataset): synthetic data generator
            n_rounds (int): sample size of batch data
            pi_e (tuple): evaluation policy. 
                          ex. ('beta', 1.0) (using beta to specify evaluation policy)
                          ex. ('function', pi(a|x), tau) (Give any function as evaluation policy. To get action_diost, we use predict_proba(tau=tau))
        """
        assert type(pi_e) == tuple, 'type of pi_e must be tuple'
        assert (pi_e[0] == 'beta') or (pi_e[0] == 'function'), 'pi_e[0] must be beta or function'

        assert dataset.random_state is None, 'set random state (int) in dataset'

        self.dataset = dataset
        self.n_rounds = n_rounds
        self.pi_e = pi_e

    def set_pass_params(self, k, alpha, tuning=False):
        """set params for pass

        Args:
            k (float): Must be >1.0.
            alpha (float): Must be >0 and <1.0.
            tuning (bool, optional): Defaults to False. If True, we tune alpha based on mse of importance weight.
        """
        self.pass_k = k
        self.pass_alpha = alpha
        self.pass_tuning = tuning

    def set_pasif_params(
        self, 
        k=0.1, 
        regularization_weight=1.0, 
        batch_size=None, 
        n_epochs=100, 
        optimizer=optim.SGD, 
        lr=0.01
        ):
        """set params for pasif

        Args:
            k (float, optional): Defaults to 0.1. Expected sample ratio of subsumpled data
            regularization_weight (float, optional): Defaults to 1.0. Coefficient of regularization in loss.
            batch_size (int, optional): Defaults to None. Batch size in training. If None, we set sample size as batch_size.
            n_epochs (int, optional): Defaults to 100. Number of epochs.
            optimizer (torch.optim, optional): Defaults to optim.SGD. Optimizer of nn.
            lr (float, optional): Defaults to 0.01. Learning rate.
        """
        self.pasif_k = k
        self.pasif_original_regularization_weight = regularization_weight
        self.pasif_batch_size = batch_size
        self.pasif_n_epochs = n_epochs
        self.pasif_optimizer = optimizer
        self.pasif_lr = lr


    def _split_data(self, batch_bandit_feedback, action_dist_by_pi_e, method='pass'):
        """split data into 2 qasi log data

        Args:
            batch_bandit_feedback (dict): batch bandit feedback
            action_dist_by_pi_e (np.array): action dist by evalation policy
            method (str, optional): Defaults to 'pass'. Must be pass or pasif.

        Returns:
            dict, dict, np.array, np.array: batch1, counterfacal action dist1, batch2, counterfacal action dist2
        """
        assert (method == 'pass') or (method == 'pasif'), 'method must be pass or pasif'
        if method == 'pass':
            data_split = DataSplittingByPass(
                batch_bandit_feedback=batch_bandit_feedback, 
                action_dist_by_pi_e=action_dist_by_pi_e, 
                k=self.pass_k, 
                alpha=self.pass_alpha, 
                random_state=self.random_state
                )
            data_split.split_data()
            batch_feedback_1, cf_action_dist_1, batch_feedback_2, cf_action_dist_2 = data_split.get_split_data()

        elif method == 'pasif':
            data_split = DataSplittingByPasif(
                batch_bandit_feedback=batch_bandit_feedback, 
                action_dist_by_pi_e=action_dist_by_pi_e, 
                random_state=self.random_state
                )
            data_split.set_params(
                k=self.pasif_k, 
                regularization_weight=self.pasif_regularization_weight, 
                batch_size=self.pasif_batch_size, 
                n_epochs=self.pasif_n_epochs, 
                optimizer=self.pasif_optimizer, 
                lr=self.pasif_lr
                )
            data_split.train_importance_fitting()
            batch_feedback_1, cf_action_dist_1, batch_feedback_2, cf_action_dist_2 = data_split.get_split_data()

            self.pasif_final_loss = data_split.final_loss
            self.pasif_final_loss_d = data_split.final_loss_d
            self.pasif_final_loss_r = data_split.final_loss_r
            self.pasif_split_info_dict = data_split.split_info_dict

        return batch_feedback_1, cf_action_dist_1, batch_feedback_2, cf_action_dist_2


    def _evaluate_estimators_single_inner_loop(
        self, 
        batch_bandit_feedback, 
        action_dist_by_pi_e, 
        method
        ):
        """For given batch data, we estimate estimator performance

        Args:
            batch_bandit_feedback (dict): batch bandit feedback
            action_dist_by_pi_e (np.array): action distribution by evaluation policy
            method (str): Name of splitting method. Must be pass or pasif.

        Returns:
            dict: key:estimator name, value:metric of estimator performance
        """
        # split data
        batch_bandit_feedback_1, action_dist_1_by_2, batch_bandit_feedback_2, action_dist_2_by_1 = \
            self._split_data(batch_bandit_feedback=batch_bandit_feedback, action_dist_by_pi_e=action_dist_by_pi_e, method=method)

        if batch_bandit_feedback_1 is None:
            # This is needed for grid search of weight
            return None
        else:
            # using split data as two log data, we use conventional estimator selection method
            conventional_method = ConventionalEstimatorSelection(
                ope_estimators=self.ope_estimators, 
                q_models=self.q_models, 
                metrics=self.metrics, 
                data_type='real', 
                random_state=self.random_state
                )
            conventional_method.set_real_data(
                batch_bandit_feedback_1=batch_bandit_feedback_1, 
                action_dist_1_by_2=action_dist_1_by_2, 
                batch_bandit_feedback_2=batch_bandit_feedback_2, 
                action_dist_2_by_1=action_dist_2_by_1, 
                evaluation_data='1'
                )
            conventional_method.evaluate_estimators(
                n_inner_bootstrap=None, 
                n_outer_bootstrap=None, 
                outer_repeat_type='bootstrap', 
                ground_truth_method='on_policy'
                )

            estimator_performance = conventional_method.get_summarized_results()
            estimator_performance = estimator_performance.set_index('estimator_name').to_dict(orient='dict')['mean '+self.metrics]

        return estimator_performance


    def _evaluate_estimators_single_outer_loop(
        self, 
        batch_bandit_feedback, 
        action_dist_by_pi_e, 
        method, 
        n_bootstrap=100, 
        ):
        """"For given batch data, we estimate estimator performance with bootstrap

        Args:
            batch_bandit_feedback (dict): batch bandit feedback
            action_dist_by_pi_e (np.array): action distribution by evaluaiton policy
            method (str): Name of splitting method.
            n_bootstrap (int, optional): Defaults to 100. The number of bootstrap sampling. If None, we use original data only once.

        Returns:
            pd.DataFrame: dataframe with columns=[estimator_name, metrics, rank]
        """

        mean_estimator_performance_dict = {}

        if n_bootstrap is None:
            estimator_performance = self._evaluate_estimators_single_inner_loop(
                batch_bandit_feedback=batch_bandit_feedback, 
                action_dist_by_pi_e=action_dist_by_pi_e, 
                method=method
                )
            mean_estimator_performance_dict.update(estimator_performance)
        else:
            for i_bootstrap in range(n_bootstrap):
                nn_output_is_nan = True
                additional_i = int(0)
                while nn_output_is_nan:
                    bootstrapped_data, bootstrapped_dist = sample_bootstrap_batch_bandit_feedback(
                        batch_bandit_feedback=batch_bandit_feedback, 
                        action_dist=action_dist_by_pi_e, 
                        sample_size_ratio=1.0,
                        random_state=self.random_state+i_bootstrap+(additional_i*n_bootstrap)
                        )
                    estimator_performance = self._evaluate_estimators_single_inner_loop(
                        batch_bandit_feedback=bootstrapped_data, 
                        action_dist_by_pi_e=bootstrapped_dist, 
                        method=method
                        )
                    if estimator_performance is None:
                        additional_i += 1
                    else:
                        nn_output_is_nan = False

                if i_bootstrap == 0:
                    mean_estimator_performance_dict.update(estimator_performance)
                else:
                    for ope_name in mean_estimator_performance_dict.keys():
                        mean_estimator_performance_dict[ope_name] += estimator_performance[ope_name]
            
            for ope_name in mean_estimator_performance_dict.keys():
                mean_estimator_performance_dict[ope_name] = mean_estimator_performance_dict[ope_name] / n_bootstrap

        mean_estimator_performance_df = pd.DataFrame({
            'estimator_name':mean_estimator_performance_dict.keys(), 
            self.metrics: mean_estimator_performance_dict.values()
            })

        estimator_rank = mean_estimator_performance_df.rank(method='min')[[self.metrics]].rename(columns={self.metrics:'rank'})
        mean_estimator_performance_df = pd.merge(mean_estimator_performance_df, estimator_rank, left_index=True, right_index=True)
            
        return mean_estimator_performance_df


    def evaluate_estimators(
        self, 
        n_inner_bootstrap, 
        n_outer_bootstrap, 
        method, 
        outer_repeat_type='bootstrap', 
        ):
        """for set data, we evaluate ope estimators with bootstrap for several times

        Args:
            n_inner_bootstrap (int): The number of bootstrap sampling in ope estimator selection. If None, we use original data only once.
            n_outer_bootstrap (int): The number of evaluation of estimators. If None, we use original data only once.
            method (str): Name of splitting method.
            outer_repeat_type (str, optional): Defaults to 'bootstrap'. If we use synthetic data, we can also use 'generation' meaning generating data for each outer loop
        """
        if self.data_type == 'synthetic':
            if self.pi_e[0] == 'beta':
                self.action_dist_by_pi_e = get_counterfactual_action_distribution(dataset=self.dataset, cf_beta= self.pi_e[1], n_rounds=self.n_rounds)
                self.batch_bandit_feedback = self.dataset.obtain_batch_bandit_feedback(n_rounds=self.n_rounds)
            elif self.pi_e[0] == 'function':
                self.batch_bandit_feedback = self.dataset.obtain_batch_bandit_feedback(n_rounds=self.n_rounds)
                self.action_dist_by_pi_e = self.pi_e[1].predict_proba(self.batch_bandit_feedback['context'], tau=self.pi_e[2])
                self.action_dist_by_pi_e = self.action_dist_by_pi_e.reshape((self.action_dist_by_pi_e.shape[0], self.action_dist_by_pi_e.shape[1], 1))

        if method == 'pass':
            if self.pass_tuning:
                print('grid search of pass_alpha start')
                n_trial = 20
                pass_alpha_result = {}
                for alpha in range(n_trial+1):
                    alpha = float(alpha/n_trial)
                    if alpha == 1.0:
                        alpha -= 0.001 # to avoid the case that pi_b includes zero (this cause actual importance weight encounter zero division)
                    data_split = DataSplittingByPass(
                        batch_bandit_feedback=self.batch_bandit_feedback, 
                        action_dist_by_pi_e=self.action_dist_by_pi_e, 
                        k=self.pass_k, 
                        alpha=alpha, 
                        random_state=self.random_state
                        )
                    data_split.split_data()
                    pass_mse = data_split.get_mse_of_importance_weight()
                    pass_alpha_result[alpha] = pass_mse
                    if alpha == 0:
                        pass_min_mse = pass_mse
                        self.pass_alpha = alpha
                    else:
                        if pass_mse < pass_min_mse:
                            pass_min_mse = pass_mse
                            self.pass_alpha = alpha
                print('grid search of pass_alpha end')
                print('result of grid search of pass_alpha (key:alpha, value:metrics):', pass_alpha_result)
                print('selected pass_alpha: ', self.pass_alpha)
            else:
                pass
        elif method == 'pasif':
            candidate_regularization_weight = [1e-1, 1e0, 1e1, 1e2, 1e3]
            if self.pasif_original_regularization_weight in [-999, -998, -997]:
                weight_grid_search_result = {} # key:weight, value:metrics to choose weight
                print('grid search of regularization_weight start')
                for temp_regularization_weight in candidate_regularization_weight:
                    self.pasif_regularization_weight = temp_regularization_weight
                    self._evaluate_estimators_single_inner_loop(
                        batch_bandit_feedback=self.batch_bandit_feedback, 
                        action_dist_by_pi_e=self.action_dist_by_pi_e, 
                        method=method
                        )
                    if self.pasif_original_regularization_weight == -999:
                        #metrics: abs diff between loss_d and loss_r
                        weight_grid_search_result[temp_regularization_weight] = abs(self.pasif_final_loss_d - self.pasif_final_loss_r)
                    elif self.pasif_original_regularization_weight == -998:
                        #metrics: abs diff between loss_d and regularization_weight * loss_r
                        weight_grid_search_result[temp_regularization_weight] = abs(self.pasif_final_loss_d - temp_regularization_weight * self.pasif_final_loss_r)
                    elif self.pasif_original_regularization_weight == -997:
                        #metrics: tuple of mean marginal_p and final loss D
                        if not(self.pasif_split_info_dict['marginal_p'] is None):
                            weight_grid_search_result[temp_regularization_weight] = (self.pasif_split_info_dict['marginal_p'].mean(), self.pasif_final_loss_d)
                        else:
                            weight_grid_search_result[temp_regularization_weight] = (None, self.pasif_final_loss_d)
                if self.pasif_original_regularization_weight in [-999, -998]:
                    # select weight with minimum metrics
                    self.pasif_regularization_weight = min(weight_grid_search_result, key=weight_grid_search_result.get)
                elif self.pasif_original_regularization_weight == -997:
                    # select weight with minimal loss D and satisfying the condition for k
                    self.pasif_regularization_weight = None
                    minimum_loss = None
                    for temp_regularization_weight, tuple_of_metrics in weight_grid_search_result.items():
                        if not(tuple_of_metrics[0] is None):
                            if (tuple_of_metrics[0] > self.pasif_k - 0.02) and (tuple_of_metrics[0] < self.pasif_k + 0.02):
                                if minimum_loss is None:
                                    minimum_loss = tuple_of_metrics[1]
                                    self.pasif_regularization_weight = temp_regularization_weight
                                else:
                                    if tuple_of_metrics[1] < minimum_loss:
                                        minimum_loss = tuple_of_metrics[1]
                                        self.pasif_regularization_weight = temp_regularization_weight
                            else:
                                pass
                        else:
                            pass
                # check results of grid search
                print('grid search of regularization_weight end')
                print('result of grid search of regularization_weight in pasif (key:weight, value:metrics):', weight_grid_search_result)
                print('selected regularization_weight: ', self.pasif_regularization_weight)
                if self.pasif_original_regularization_weight == -997:
                    assert not(self.pasif_regularization_weight is None), 'There were no candidates of regularization_weight that met the conditions for k'
            else:
                self.pasif_regularization_weight = self.pasif_original_regularization_weight

        if n_outer_bootstrap is None:
            self.all_result = self._evaluate_estimators_single_outer_loop(
                batch_bandit_feedback=self.batch_bandit_feedback, 
                action_dist_by_pi_e=self.action_dist_by_pi_e, 
                method=method, 
                n_bootstrap=n_inner_bootstrap
                )
            self.all_result['outer_iteration'] = 0
        else:
            for i_outer in range(n_outer_bootstrap):
                if outer_repeat_type == 'bootstrap':
                    bootstrapped_data, bootstrapped_dist = sample_bootstrap_batch_bandit_feedback(
                        batch_bandit_feedback=self.batch_bandit_feedback, 
                        action_dist=self.action_dist_by_pi_e, 
                        sample_size_ratio=1.0,
                        random_state=self.random_state+i_outer
                        )
                elif outer_repeat_type == 'generation':
                    if self.pi_e[0] == 'beta':
                        bootstrapped_dist = get_counterfactual_action_distribution(dataset=self.dataset, cf_beta= self.pi_e[1], n_rounds=self.n_rounds)
                        bootstrapped_data = self.dataset.obtain_batch_bandit_feedback(n_rounds=self.n_rounds)
                    elif self.pi_e[0] == 'function':
                        bootstrapped_data = self.dataset.obtain_batch_bandit_feedback(n_rounds=self.n_rounds)
                        bootstrapped_dist = self.pi_e[1].predict_proba(bootstrapped_data['context'], tau=self.pi_e[2])
                        bootstrapped_dist = bootstrapped_dist.reshape((bootstrapped_dist.shape[0], bootstrapped_dist.shape[1], 1))

                i_outer_result = self._evaluate_estimators_single_outer_loop(
                    batch_bandit_feedback=bootstrapped_data, 
                    action_dist_by_pi_e=bootstrapped_dist, 
                    method=method, 
                    n_bootstrap=n_inner_bootstrap
                    )
                i_outer_result['outer_iteration'] = i_outer
                if i_outer == 0:
                    self.all_result = i_outer_result
                else:
                    self.all_result = pd.concat([self.all_result, i_outer_result])


        self.all_result = self.all_result[['outer_iteration', 'estimator_name', self.metrics, 'rank']]
        self.summarized_result = pd.DataFrame(columns=['estimator_name','mean '+self.metrics, 'stdev', '95%CI(upper)', '95%CI(lower)'])
        for estimator_name in self.all_result['estimator_name'].unique():
            summarized_result = [estimator_name]
            summarized_result.append(self.all_result[self.metrics][self.all_result['estimator_name']==estimator_name].mean())
            if self.all_result['outer_iteration'].max() > 0:
                summarized_result.append(self.all_result[self.metrics][self.all_result['estimator_name']==estimator_name].std())
                t_dist = stats.t(loc=summarized_result[1],
                                scale=stats.sem(self.all_result[self.metrics][self.all_result['estimator_name']==estimator_name]),
                                df=len(self.all_result[self.metrics][self.all_result['estimator_name']==estimator_name])-1)
                bottom, up = t_dist.interval(alpha=0.95)
                summarized_result.append(up)
                summarized_result.append(bottom)
            else:
                summarized_result.append(None)
                summarized_result.append(None)
                summarized_result.append(None)
            self.summarized_result = self.summarized_result.append({
                'estimator_name':summarized_result[0],
                'mean '+self.metrics:summarized_result[1], 
                'stdev':summarized_result[2], 
                '95%CI(upper)':summarized_result[3], 
                '95%CI(lower)':summarized_result[4]
                }, ignore_index=True)

        
        estimator_rank = self.summarized_result.rank(method='min')[['mean '+self.metrics]].rename(columns={'mean '+self.metrics:'rank'})
        self.summarized_result = pd.merge(self.summarized_result, estimator_rank, left_index=True, right_index=True)
            

    def get_all_results(self):
        """get all evaluation results (all outer lopp results)

        Returns:
            pd.DataFrame: results for all outer loop
        """
        return self.all_result

    def get_summarized_results(self):
        """get summarized evaluation results

        Returns:
            pd.DataFrame: results (mean results for outer loops)
        """
        return self.summarized_result

    def get_best_estimator(self):
        """get best ope estimator besed on mean metric

        Returns:
            obp.ope, sklearn-predictor: best ope estimator and q model
        """

        # define best estimator
        best_estimator_name = self.summarized_result['estimator_name'][self.summarized_result['rank']==1].values[0]
        idx = best_estimator_name.find('_qmodel_')
        if idx == -1:
            best_ope_name = best_estimator_name
        else:
            best_ope_name = best_estimator_name[:idx]

        for estimator in self.ope_estimators:
            if estimator.estimator_name == best_ope_name:
                best_estimator = estimator
        best_q_model = self.q_models[0]
        for q_model in self.q_models:
            if str(q_model) in best_estimator_name:
                best_q_model = q_model

        return best_estimator, best_q_model


    def visualize_results(self, show=False, save_path=None):
        """show bar-graph of evaluation metrics (with 95% CI)

        Args:
            show (bool, optional): Defaults to False. If True, show figure.
            save_path (str, optional): Defaults to None. If None, we do not save figure. If not None, we save figure in save_path.
        """

        # error range of 95% CI
        err_range = []
        err_range.append(self.summarized_result['mean '+self.metrics] - self.summarized_result['95%CI(lower)'])
        err_range.append(self.summarized_result['95%CI(upper)'] - self.summarized_result['mean '+self.metrics])
        
        # plot the result
        plt.figure()
        plt.bar(
            self.summarized_result['estimator_name'], 
            self.summarized_result['mean '+self.metrics], 
            yerr=err_range
            )
        plt.xlabel('OPE estimators')
        plt.ylabel(self.metrics)
        if show:
            plt.show()
        if not(save_path is None):
            plt.savefig(save_path)

