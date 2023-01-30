from cmath import log
from copyreg import pickle
import imp
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import random
import copy
import glob
import pickle
import csv
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LogisticRegression

import torch.optim as optim

import obp
from obp.dataset import SyntheticBanditDataset, logistic_reward_function, linear_reward_function, linear_behavior_policy
from obp.ope import OffPolicyEvaluation, RegressionModel
from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting as IPW, DirectMethod as DM, DoublyRobust as DR, RegressionModel
from obp.ope import SelfNormalizedInverseProbabilityWeighting as SNIPW, SwitchDoublyRobust as SwitchDR, DoublyRobustWithShrinkage as DRos
from obp.ope import SelfNormalizedDoublyRobust as SNDR
from obp.ope import InverseProbabilityWeightingTuning as IPWTuning, DoublyRobustTuning as DRTuning, SwitchDoublyRobustTuning as SwitchDRTuning, DoublyRobustWithShrinkageTuning as DRosTuning
from obp.ope import SubGaussianInverseProbabilityWeightingTuning as SGIPWTuning, SubGaussianDoublyRobustTuning as SGDRTuning

import os
import sys
if os.path.dirname(__file__) == '':
    sys.path.append('../../src')
else:
    sys.path.append(os.path.dirname(__file__) + '/../../src')
from data.multiple_beta_synthetic_data import MultipleBetaSyntheticData
from estimator_selection.conventional_estimator_selection import ConventionalEstimatorSelection
from policy_selection.conventional_policy_selection import ConventionalPolicySelection
from policy_selection.proposed_policy_selection import ProposedPolicySelection
from evaluation.evaluation_metric import calculate_relative_regret_e, calculate_rank_correlation_coefficient_e
from evaluation.evaluation_metric import calculate_relative_regret_p, calculate_rank_correlation_coefficient_p

import warnings
warnings.simplefilter('ignore')


class LearnedBehaviorPolicy:
    """class to use trained policy for SyntheticBanditDataset
    """
    def __init__(self, model):
        """set trained policy model

        Args:
            model (obp.policy): trained policy model
        """
        self.model = model

    def behavior_policy_function_predict_proba(self, context, action_context, random_state):
        """This can be used for behavior policy of SyntheticBanditDataset

        Args:
            context (np.array): context
            action_context (np.array): Conveniently set up for use within the SyntheticBanditDataset.
            random_state (int): Conveniently set up for use within the SyntheticBanditDataset.
        Returns:
            pd.DataFrame, dict: dataframe with columns=[policy_name, estimator_name, estimated_policy_value, rank], and estimator selection result for each evaluation policy
        """
        predicted_action_dist = self.model.predict_score(context)
        predicted_action_dist = predicted_action_dist[:,:,0]
        return predicted_action_dist


class EvaluationOfSelectionMethodWithSyntheticData:
    """Using synthetic data, we evaluate and compare conventional/proposed estimator/policy selection method.
    """

    def __init__(
        self, 
        ope_estimators, 
        q_models, 
        estimator_selection_metrics='mse', 
        n_actions=5, 
        dim_context=5, 
        beta_1=3, 
        beta_2=7, 
        reward_type='binary', 
        reward_function=logistic_reward_function, 
        n_rounds_1=1000, 
        n_rounds_2=1000, 
        test_ratio=0.5, 
        pi_e={'beta_0.0':('beta', 0.0), 'beta_5.0':('beta', 5.0), 'beta_10.0':('beta', 10.0)}, 
        n_data_generation=10, 
        random_state=None
        ):
        """set basic settings

        Args:
            ope_estimators (list): list of candidate estimators
            q_models (list): list of reward estimators used in model-depending estimators
            estimator_selection_metrics (str, optional): _description_. Defaults to 'mse'.
            n_actions (int, optional): Defaults to 5. Number of actions.
            dim_context (int, optional): Defaults to 5. Dim of context.
            beta_1 (int, optional): Defaults to 3. Beta for partial log data.
            beta_2 (int, optional): Defaults to 7. Beta for partial log data.
            reward_type (str, optional): Defaults to 'binary'. binary or continuous.
            reward_function (_type_, optional): Defaults to logistic_reward_function. mean reward function.
            n_rounds_1 (int, optional): Defaults to 1000. sample size of partial data 1.
            n_rounds_2 (int, optional): Defaults to 1000. sample size of partial data 2.
            test_ratio (float, optional): Defaults to 0.5. Ratio of test set.
                                          If None, we use whole data for both estimator selection and policy selection (valid when outer_loop_type='generation').
            pi_e (dict, optional): dictionary of evaluation policies. keys: name of policy (str). values: tuple including info of evaluation policy.
                                   ex. {'beta_1.0':('beta', 1.0), 'function_1':('function', pi(a|x), tau)} 
                                   * ('beta', 1.0) (using beta to specify evaluation policy)
                                   * ('function', pi(a|x), tau) (Give any function as evaluation policy. To get action_diost, we use predict_proba(tau=tau))
            n_data_generation (int, optional): Defaults to 10. The number of policy selection. 
            random_state (int, optional): Defaults to None.
        """
        self.ope_estimators = ope_estimators
        self.q_models = q_models
        self.estimator_selection_metrics = estimator_selection_metrics
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.reward_type = reward_type
        self.reward_function = reward_function
        self.n_rounds_1 = n_rounds_1
        self.n_rounds_2 = n_rounds_2
        self.test_ratio = test_ratio
        self.pi_e = pi_e
        self.n_data_generation = n_data_generation
        if random_state is None:
            self.random_state = random.randint(1,10000000)
        else:
            self.random_state = random_state


    def set_conventional_method_params(
        self, 
        n_inner_bootstrap=100, 
        evaluation_data='partial_random'
        ):
        """set params for convantional estimator/policy selection

        Args:
            n_inner_bootstrap (int, optional): Defaults to 100. The number of bootstrap sampling in ope estimator selection.
            evaluation_data (str, optional): Defaults to 'partial_random'. Must be '1' or '2' or 'random' or 'partial_random'. 
                                             Which data (behavior policy) to consider as evaluation policy.
                                             'partial_random' means that we use fixed data as evalu policy in bootstrap, but no fixed in outer loop in estimator selection.
        """
        self.c_n_inner_bootstrap = n_inner_bootstrap
        self.c_evaluation_data = evaluation_data

    def set_proposed_method_params(
        self, 
        method_name, 
        method_params, 
        n_inner_bootstrap=100, 
        ):
        """set params for proposed estimator/policy selection

        Args:
            method_name (str)): name of data splitting method. Must be pass or pasif.
            method_params (dict): params for data splitting. 
                                  if you use pass, this is dict of "key:name of param, value:value of param".
                                  ex (pass). {'k':2.0, 'alpha':1.0, 'tuning':False}
                                  if you use pasif, this is dict of "key:name of policy, value:dict (key:name of param, value:value of param)".
                                  ex (pasif). { 'beta_1.0':{'k':0.1, 'regularization_weight':0.1, 'batch_size':2000, 'n_epochs':10000, 'optimizer':optim.SGD, 'lr':lr }  }
            n_inner_bootstrap (int, optional): Defaults to 100. The number of bootstrap sampling in ope estimator selection.
        """
        self.p_n_inner_bootstrap = n_inner_bootstrap
        self.p_method_name = method_name
        self.p_method_params = method_params

    def set_ground_truth(self, n_sampling):
        """set ground truth of estimator/policy selection

        Args:
            n_sampling (int): Number of sampling of log data to calculate true MSE
        """
        self.estimator_selection_gt = {} 
        self.policy_selection_gt = pd.DataFrame(columns=['policy_name', 'policy_value'])

        for policy_name, pi_e in self.pi_e.items():
            # {'beta_1.0':('beta', 1.0), 'function_1':('function', pi(a|x), tau)} 
            # ground truth of estimator performance
            conventinal_estimator_selection = ConventionalEstimatorSelection(
                ope_estimators=self.ope_estimators, 
                q_models=self.q_models, 
                metrics=self.estimator_selection_metrics, 
                data_type='synthetic', 
                random_state=self.random_state
                )
            if pi_e[0] == 'beta':
                dataset_1 = SyntheticBanditDataset(
                    n_actions=self.n_actions,
                    dim_context=self.dim_context,
                    beta=pi_e[1], 
                    reward_type=self.reward_type,
                    reward_function=self.reward_function,
                    behavior_policy_function=None,
                    random_state=self.random_state
                    )
            elif pi_e[0] == 'function':
                learned_policy = LearnedBehaviorPolicy(pi_e[1])
                dataset_1 = SyntheticBanditDataset(
                    n_actions=self.n_actions,
                    dim_context=self.dim_context,
                    beta=1.0/pi_e[2], #1.0/tau == beta
                    reward_type=self.reward_type,
                    reward_function=self.reward_function,
                    behavior_policy_function=learned_policy.behavior_policy_function_predict_proba,
                    random_state=self.random_state
                    )
            dataset_2 = MultipleBetaSyntheticData(
                n_actions=self.n_actions,
                dim_context=self.dim_context,
                beta=[self.beta_1, self.beta_2], 
                reward_type=self.reward_type,
                reward_function=self.reward_function,
                behavior_policy_function=None,
                random_state=self.random_state
                )
            if self.test_ratio is None:
                conventinal_estimator_selection.set_synthetic_data(
                    dataset_1=dataset_1, 
                    n_rounds_1=1000000, 
                    dataset_2=dataset_2, 
                    n_rounds_2=self.n_rounds_1+self.n_rounds_2, 
                    evaluation_data='1'
                    )
            else:
                conventinal_estimator_selection.set_synthetic_data(
                    dataset_1=dataset_1, 
                    n_rounds_1=1000000, 
                    dataset_2=dataset_2, 
                    n_rounds_2=int(self.test_ratio(self.n_rounds_1+self.n_rounds_2)), 
                    evaluation_data='1'
                    )
            conventinal_estimator_selection.evaluate_estimators(
                n_inner_bootstrap=None, 
                n_outer_bootstrap=n_sampling, 
                outer_repeat_type='generation', 
                ground_truth_method='ground-truth'
                )
            estimator_performance = conventinal_estimator_selection.get_summarized_results()
            estimator_performance = \
                estimator_performance[['estimator_name','mean '+self.estimator_selection_metrics, 'rank']].rename(columns={'mean '+self.estimator_selection_metrics:self.estimator_selection_metrics})
            self.estimator_selection_gt[policy_name] = estimator_performance

            # ground truth of evaluation policy
            batch_with_large_n = dataset_1.obtain_batch_bandit_feedback(n_rounds=1000000)
            policy_value = dataset_1.calc_ground_truth_policy_value(
                expected_reward=batch_with_large_n['expected_reward'],
                action_dist=batch_with_large_n['pi_b']
                )
            self.policy_selection_gt = self.policy_selection_gt.append({
                'policy_name':policy_name,
                'policy_value':policy_value
                }, ignore_index=True)

        policy_rank = self.policy_selection_gt.rank(method='min', ascending=False)[['policy_value']].rename(columns={'policy_value':'rank'})
        self.policy_selection_gt = pd.merge(self.policy_selection_gt, policy_rank, left_index=True, right_index=True)
            
    def evaluate_conventional_selection_method(self):
        """evaluate conventional estimator/policy selection
        """
        # estimator/plicy selection
        dataset_1 = SyntheticBanditDataset(
            n_actions=self.n_actions,
            dim_context=self.dim_context,
            beta=self.beta_1, 
            reward_type=self.reward_type,
            reward_function=self.reward_function,
            behavior_policy_function=None,
            random_state=self.random_state
            )
        dataset_2 = SyntheticBanditDataset(
            n_actions=self.n_actions,
            dim_context=self.dim_context,
            beta=self.beta_2, 
            reward_type=self.reward_type,
            reward_function=self.reward_function,
            behavior_policy_function=None,
            random_state=self.random_state
            )
        self.conventional_policy_selection = ConventionalPolicySelection(
            ope_estimators=self.ope_estimators, 
            q_models=self.q_models, 
            estimator_selection_metrics=self.estimator_selection_metrics, 
            data_type='synthetic', 
            random_state=self.random_state
            )
        self.conventional_policy_selection.set_synthetic_data(
            dataset_1=dataset_1, 
            n_rounds_1=self.n_rounds_1, 
            dataset_2=dataset_2, 
            n_rounds_2=self.n_rounds_2, 
            pi_e=self.pi_e, 
            evaluation_data='partial_random'
            )
        self.conventional_policy_selection.evaluate_policies(
            n_inner_bootstrap=self.c_n_inner_bootstrap, 
            n_outer_loop=self.n_data_generation, 
            outer_loop_type='generation', 
            test_ratio=self.test_ratio
            )
        self.c_estimator_selection_result = \
            self.conventional_policy_selection.get_all_estimator_selection_results()
        for pol_name, result_df in self.c_estimator_selection_result.items():
            self.c_estimator_selection_result[pol_name] = result_df.rename(columns={'mean '+self.estimator_selection_metrics:'estimated '+self.estimator_selection_metrics})
        self.c_policy_selection_result = self.conventional_policy_selection.get_all_results()
        # evaluation of estimator/plicy selection
        self.evaluation_of_c_estimator_selection = {} # key:policy_name, value:dataframe
        for policy_name, estimator_selection_result in self.c_estimator_selection_result.items():
            evaluation_of_c_estimator_selection = pd.DataFrame(columns=['outer_iteration', 'relative_regret', 'rank_correlation_coefficient'])
            for outer_loop in range(self.n_data_generation):
                predict_data = estimator_selection_result[estimator_selection_result['outer_iteration']==outer_loop]
                true_data = self.estimator_selection_gt[policy_name]
                relative_regret_e = calculate_relative_regret_e(true_data=true_data, estimated_data=predict_data, estimator_selection_metrics=self.estimator_selection_metrics)
                rank_cc_e = calculate_rank_correlation_coefficient_e(true_data=true_data, estimated_data=predict_data)
                evaluation_of_c_estimator_selection = evaluation_of_c_estimator_selection.append({
                    'outer_iteration':outer_loop, 
                    'relative_regret':relative_regret_e, 
                    'rank_correlation_coefficient':rank_cc_e
                    }, ignore_index=True)
            self.evaluation_of_c_estimator_selection[policy_name] = evaluation_of_c_estimator_selection

        self.evaluation_of_c_policy_selection = pd.DataFrame(columns=['outer_iteration', 'relative_regret', 'rank_correlation_coefficient'])
        for outer_loop in range(self.n_data_generation):
            predict_data = self.c_policy_selection_result[self.c_policy_selection_result['outer_iteration']==outer_loop]
            true_data = self.policy_selection_gt
            relative_regret_p = calculate_relative_regret_p(true_data=true_data, estimated_data=predict_data)
            rank_cc_p = calculate_rank_correlation_coefficient_p(true_data=true_data, estimated_data=predict_data)
            self.evaluation_of_c_policy_selection = self.evaluation_of_c_policy_selection.append({
                'outer_iteration':outer_loop, 
                'relative_regret':relative_regret_p, 
                'rank_correlation_coefficient':rank_cc_p
                }, ignore_index=True)


    def evaluate_proposed_selection_method(self):
        """evaluate proposed estimator/policy selection
        """
        # estimator/plicy selection
        dataset = MultipleBetaSyntheticData(
            n_actions=self.n_actions,
            dim_context=self.dim_context,
            beta=[self.beta_1, self.beta_2], 
            reward_type=self.reward_type,
            reward_function=self.reward_function,
            behavior_policy_function=None,
            random_state=self.random_state
            )
        self.proposed_policy_selection = ProposedPolicySelection(
            ope_estimators=self.ope_estimators, 
            q_models=self.q_models, 
            estimator_selection_metrics=self.estimator_selection_metrics, 
            data_type='synthetic', 
            random_state=self.random_state
            )
        self.proposed_policy_selection.set_synthetic_data(
            dataset=dataset, 
            n_rounds=self.n_rounds_1+self.n_rounds_2, 
            pi_e=self.pi_e
            )
        if self.p_method_name == 'pass':
            self.proposed_policy_selection.set_pass_params(
                k=self.p_method_params['k'], 
                alpha=self.p_method_params['alpha'], 
                tuning=self.p_method_params['tuning']
                )
        elif self.p_method_name == 'pasif':
            self.proposed_policy_selection.set_pasif_params(
                params=self.p_method_params
                )
        self.proposed_policy_selection.evaluate_policies(
            method=self.p_method_name, 
            n_inner_bootstrap=self.p_n_inner_bootstrap, 
            n_outer_loop=self.n_data_generation, 
            outer_loop_type='generation', 
            test_ratio=self.test_ratio
            )
        self.p_estimator_selection_result = \
            self.proposed_policy_selection.get_all_estimator_selection_results()
        for pol_name, result_df in self.p_estimator_selection_result.items():
            self.p_estimator_selection_result[pol_name] = result_df.rename(columns={'mean '+self.estimator_selection_metrics:'estimated '+self.estimator_selection_metrics})
        self.p_policy_selection_result = self.proposed_policy_selection.get_all_results()
        # evaluation of estimator/plicy selection
        self.evaluation_of_p_estimator_selection = {} # key:policy_name, value:dataframe
        for policy_name, estimator_selection_result in self.p_estimator_selection_result.items():
            evaluation_of_p_estimator_selection = pd.DataFrame(columns=['outer_iteration', 'relative_regret', 'rank_correlation_coefficient'])
            for outer_loop in range(self.n_data_generation):
                predict_data = estimator_selection_result[estimator_selection_result['outer_iteration']==outer_loop]
                true_data = self.estimator_selection_gt[policy_name]
                relative_regret_e = calculate_relative_regret_e(true_data=true_data, estimated_data=predict_data, estimator_selection_metrics=self.estimator_selection_metrics)
                rank_cc_e = calculate_rank_correlation_coefficient_e(true_data=true_data, estimated_data=predict_data)
                evaluation_of_p_estimator_selection = evaluation_of_p_estimator_selection.append({
                    'outer_iteration':outer_loop, 
                    'relative_regret':relative_regret_e, 
                    'rank_correlation_coefficient':rank_cc_e
                    }, ignore_index=True)
            self.evaluation_of_p_estimator_selection[policy_name] = evaluation_of_p_estimator_selection

        self.evaluation_of_p_policy_selection = pd.DataFrame(columns=['outer_iteration', 'relative_regret', 'rank_correlation_coefficient'])
        for outer_loop in range(self.n_data_generation):
            predict_data = self.p_policy_selection_result[self.p_policy_selection_result['outer_iteration']==outer_loop]
            true_data = self.policy_selection_gt
            relative_regret_p = calculate_relative_regret_p(true_data=true_data, estimated_data=predict_data)
            rank_cc_p = calculate_rank_correlation_coefficient_p(true_data=true_data, estimated_data=predict_data)
            self.evaluation_of_p_policy_selection = self.evaluation_of_p_policy_selection.append({
                'outer_iteration':outer_loop, 
                'relative_regret':relative_regret_p, 
                'rank_correlation_coefficient':rank_cc_p
                }, ignore_index=True)


    def get_mean_evaluation_results_of_estimator_selection(self):
        """get mean evaluation results of conventional/proposed estimator selection

        Returns:
            dict, dict, dict, dict: results for conventional/proposed method. key:policy name. value:dict of mean or std evaluation metrics.
        """
        c_mean_results = {}
        for policy_name, estimator_selection_result in self.evaluation_of_c_estimator_selection.items():
            c_mean_results[policy_name] = {
                'relative_regret':estimator_selection_result['relative_regret'].mean(), 
                'rank_correlation_coefficient':estimator_selection_result['rank_correlation_coefficient'].mean()
                }
        p_mean_results = {}
        for policy_name, estimator_selection_result in self.evaluation_of_p_estimator_selection.items():
            p_mean_results[policy_name] = {
                'relative_regret':estimator_selection_result['relative_regret'].mean(), 
                'rank_correlation_coefficient':estimator_selection_result['rank_correlation_coefficient'].mean()
                }
        c_std_results = {}
        for policy_name, estimator_selection_result in self.evaluation_of_c_estimator_selection.items():
            c_std_results[policy_name] = {
                'relative_regret':estimator_selection_result['relative_regret'].std(), 
                'rank_correlation_coefficient':estimator_selection_result['rank_correlation_coefficient'].std()
                }
        p_std_results = {}
        for policy_name, estimator_selection_result in self.evaluation_of_p_estimator_selection.items():
            p_std_results[policy_name] = {
                'relative_regret':estimator_selection_result['relative_regret'].std(), 
                'rank_correlation_coefficient':estimator_selection_result['rank_correlation_coefficient'].std()
                }

        return c_mean_results, p_mean_results, c_std_results, p_std_results

    def get_all_evaluation_results_of_estimator_selection(self):
        """get all evaluation results of conventional/proposed estimator selection

        Returns:
            dict, dict: results for conventional/proposed method. key:policy name. value:dataframe of evaluation results.
        """
        return self.evaluation_of_c_estimator_selection, self.evaluation_of_p_estimator_selection



    def get_mean_evaluation_results_of_policy_selection(self):
        """get mean evaluation results of conventional/proposed policy selection

        Returns:
            dict, dict, dict, dict: results for conventional/proposed method. key:name fo evaluation metrics. value:mean or std of result.
                                    c_mean_results, p_mean_results, c_std_results, p_std_results.
        """
        c_mean_results = {
            'relative_regret':self.evaluation_of_c_policy_selection['relative_regret'].mean(), 
            'rank_correlation_coefficient':self.evaluation_of_c_policy_selection['rank_correlation_coefficient'].mean()
            }
        p_mean_results = {
            'relative_regret':self.evaluation_of_p_policy_selection['relative_regret'].mean(), 
            'rank_correlation_coefficient':self.evaluation_of_p_policy_selection['rank_correlation_coefficient'].mean()
            }
        c_std_results = {
            'relative_regret':self.evaluation_of_c_policy_selection['relative_regret'].std(), 
            'rank_correlation_coefficient':self.evaluation_of_c_policy_selection['rank_correlation_coefficient'].std()
            }
        p_std_results = {
            'relative_regret':self.evaluation_of_p_policy_selection['relative_regret'].std(), 
            'rank_correlation_coefficient':self.evaluation_of_p_policy_selection['rank_correlation_coefficient'].std()
            }

        return c_mean_results, p_mean_results, c_std_results, p_std_results

    def get_all_evaluation_results_of_policy_selection(self):
        """get all evaluation results of conventional/proposed policy selection

        Returns:
            pd.DataFrame, pd.DataFrame: results for conventional/proposed method.
        """
        return self.evaluation_of_c_policy_selection, self.evaluation_of_p_policy_selection







if __name__ == '__main__':
    
    def mk_log_dir(dir_name):
        dir_list = glob.glob('./log/*')
        same_dir_count =0
        for existing_dir in dir_list:
            if dir_name in existing_dir:
                same_dir_count += 1
        if same_dir_count == 0:
            mk_dir_name = dir_name
        else:
            mk_dir_name = dir_name + '_ver' + str(same_dir_count)
        os.mkdir(mk_dir_name)
        return mk_dir_name

    def save_config(dir_name, args, ope_estimators, hypara_dict, q_models):
        file_path = dir_name + '/config.csv'
        with open(file_path, 'w') as f:
            writer = csv.writer(f)

            writer.writerow(['arguments'])
            dict_args = vars(args)
            for arg_key in dict_args:
                writer.writerow([arg_key, dict_args[arg_key]])

            writer.writerow([])
            writer.writerow(['ope_estimators'])
            for estimator in ope_estimators:
                writer.writerow([str(estimator)])

            writer.writerow([])
            writer.writerow(['hypara_space'])
            for hypara_name in hypara_dict.keys():
                writer.writerow([hypara_name, hypara_dict[hypara_name]])

            writer.writerow([])
            writer.writerow(['q_models'])
            for q_model in q_models:
                writer.writerow([str(q_model)])


    def save_summarized_results(dir_name, evaluation_of_selection_method):
        file_path = dir_name + '/summarized_results.csv'
        with open(file_path, 'w') as f:
            writer = csv.writer(f)

            writer.writerow(['estimator_selection'])
            writer.writerow(['pi_e', 'metric', 'conventional_mean', 'conventional_std', 'proposed_mean', 'proposed_std'])
            c_mean_results, p_mean_results, c_std_results, p_std_results = evaluation_of_selection_method.get_mean_evaluation_results_of_estimator_selection()
            for policy_name in c_mean_results.keys():
                writer.writerow([
                    policy_name, 
                    'relative_regret', 
                    c_mean_results[policy_name]['relative_regret'], 
                    c_std_results[policy_name]['relative_regret'], 
                    p_mean_results[policy_name]['relative_regret'], 
                    p_std_results[policy_name]['relative_regret']
                    ])
                writer.writerow([
                    policy_name, 
                    'rank_correlation_coefficient', 
                    c_mean_results[policy_name]['rank_correlation_coefficient'], 
                    c_std_results[policy_name]['rank_correlation_coefficient'], 
                    p_mean_results[policy_name]['rank_correlation_coefficient'], 
                    p_std_results[policy_name]['rank_correlation_coefficient']
                    ])

            writer.writerow([])
            writer.writerow(['policy_selection'])
            writer.writerow(['metric', 'conventional_mean', 'conventional_std', 'proposed_mean', 'proposed_std'])
            c_mean_results, p_mean_results, c_std_results, p_std_results = evaluation_of_selection_method.get_mean_evaluation_results_of_policy_selection()
            writer.writerow([
                'relative_regret', 
                c_mean_results['relative_regret'], 
                c_std_results['relative_regret'], 
                p_mean_results['relative_regret'], 
                p_std_results['relative_regret']
                ])
            writer.writerow([
                'rank_correlation_coefficient', 
                c_mean_results['rank_correlation_coefficient'], 
                c_std_results['rank_correlation_coefficient'], 
                p_mean_results['rank_correlation_coefficient'], 
                p_std_results['rank_correlation_coefficient']
                ])


    def save_figure(dir_name, metric_name, pi_e, evaluation_of_selection_method):
        merged_df = pd.DataFrame(index=[], columns=['method', 'beta (evaluation policy)', metric_name])
        evaluation_of_c_estimator_selection, evaluation_of_p_estimator_selection = evaluation_of_selection_method.get_all_evaluation_results_of_estimator_selection()
        for policy_name, beta_e in pi_e.items():
            if beta_e[0] == 'beta':
                c_df = pd.DataFrame({'method':'conventional', 'beta (evaluation policy)':beta_e[1], metric_name:evaluation_of_c_estimator_selection[policy_name][metric_name].values})
                p_df = pd.DataFrame({'method':'proposed', 'beta (evaluation policy)':beta_e[1], metric_name:evaluation_of_p_estimator_selection[policy_name][metric_name].values})
                merged_df = pd.concat([merged_df, c_df, p_df])
        merged_df = merged_df.reset_index()
        file_path = dir_name + '/estimator_selection_' + metric_name + '.png'
        plt.figure()
        sns.set()
        sns.lineplot(x='beta (evaluation policy)', y=metric_name, hue='method', data=merged_df) # default 95% confidence interval, https://qiita.com/ninomiyt/items/cda7ee0b940dd461cd09
        plt.savefig(file_path)


    def convert_sec_to_hms(sec):
        hour = int(sec/3600)
        minute = int((sec - 3600*hour) / 60)
        new_sec = sec - 3600*hour - 60*minute
        return hour, minute, new_sec



    processing_time_list = []

    # get argument
    parser = argparse.ArgumentParser()
    # number of actions
    parser.add_argument('--n_actions', type=int, default=5)
    # number of dimensions of context
    parser.add_argument('--dim_context', type=int, default=5)
    # type of reward
    parser.add_argument('--reward_type', type=str, choices=['binary', 'continuous'], default='binary')
    # type of reward function
    parser.add_argument('--reward_function', type=int, choices=[0, 1], help='0:logistic_reward_function, 1:linear_reward_function', default=0)
    reward_function_dict = {0:logistic_reward_function, 1:linear_reward_function}
    # beta of behavior policy 1
    parser.add_argument('--beta_1', type=float, help='beta_1 for conventional method', default=0.2)
    # beta of behavior policy 2
    parser.add_argument('--beta_2', type=float, help='beta_2 for conventional method', default=1.0)
    # sample size of log data 1
    parser.add_argument('--n1', type=int, help='n_rounds for beta_1', default=1000)
    # sample size of log data 2
    parser.add_argument('--n2', type=int, help='n_rounds for beta_2', default=1000)
    # random state
    parser.add_argument('--random_state', type=int, default=1)
    # beta of evaluation policy
    parser.add_argument('--beta_list_for_pi_e', required=True, nargs="*", type=float, default=[-100, -10, -1, -0.5, 0, 0.5, 1, 10, 100]) #https://qiita.com/hook125/items/0ffc6b9391ccb0abcd52
    # type of evaluation policy
    # 0:No model, 1:IPWLearner with LogisticRegression, 2:IPWLearner with RandomForest, 3:QLearner with LogisticRegression or RidgeRegression, 4:QLearner with RandomForest
    # if you want to use trained model (1-4), please run model/train_opl_model.py with same n_actions, dim_context, reward_function and random_state before runing this code.
    parser.add_argument('--model_list_for_pi_e', required=True, nargs="*", type=int, default=[0, 0, 0, 0, 0, 0, 1, 1, 2])
    # metric to evaluate the accuracy of estimators
    parser.add_argument('--metric', type=str, choices=['mse', 'mean relative-ee'], default='mse')
    # number of bootstrap in estimator selection procedure
    parser.add_argument('--n_bootstrap', type=int, help='number of bootstrap in estimator selection', default=100)
    # number of data generations of log data
    parser.add_argument('--n_data_generation', type=int, help='number of policy selection phase', default=10)
    # name of proposed method (please set pasif. pass and pass_tuning is old method)
    parser.add_argument('--proposed_method', type=str, choices=['pasif', 'pass', 'pass_tuning'], default='pasif')
    # parameter k in pass
    parser.add_argument('--pass_k', type=float, help='k for pass', default=2.0)
    # parameter alpha in pass
    parser.add_argument('--pass_alpha', type=float, help='alpha for pass', default=1.0)
    # parameter k in pass_tuning
    parser.add_argument('--pass_tuning_k', type=float, help='k for pass_tuning', default=2.0)
    # parameter k in pasif (for all evaluation policy)
    parser.add_argument('--pasif_k', nargs="*", type=float, help='k for pasif', default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # parameter lambda (regularization weight) in pasif (for all evaluation policy)
    # -997 means the automatic search of better regularization weight used in the paper
    parser.add_argument('--pasif_regularization_weight', nargs="*", type=float, help='regularization_weight for pasif. -999/-998/-997 means automatic search.', default=[100.0, 100.0, 10.0, 1.0, 1.0, 1.0, 10.0, 100.0, 100.0])
    # batch size of importance fitting
    parser.add_argument('--pasif_batch_size', nargs="*", type=int, help='batch_size for pasif', default=[2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000])
    # number of epochs of importance fitting
    parser.add_argument('--pasif_n_epochs', nargs="*", type=int, help='n_epochs for pasif', default=[10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])
    # optimiser of importance fitting
    parser.add_argument('--pasif_optimizer', nargs="*", type=int, help='optimizer for pasif, 0:SGD, 1:Adam', default=[0, 0, 0, 0, 0, 0, 0, 0, 0])
    pasif_optimizer_dict = {0:optim.SGD, 1:optim.Adam}
    # learning rate of importance fitting
    parser.add_argument('--pasif_lr', nargs="*", type=float, help='learning rate for pasif', default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    # number of data generations to calculate the ground truth of estimator selection
    parser.add_argument('--gt_n_sampling', type=int, help='number of sampling to calculate ground truth', default=100)
    # directory to save the results
    parser.add_argument('--save_dir', type=str, help='dir path to save results', default='.')
    arguments = parser.parse_args()


    # set basic info
    ipw_lambda_list = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, np.inf]
    dr_lambda_list = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, np.inf]
    switchdr_lambda_list = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, np.inf]
    dros_lambda_list = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, np.inf] # following the config of https://arxiv.org/pdf/2108.13703.pdf
    sgipw_lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sgdr_lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    hypara_dict = {
        'ipw_lambda':ipw_lambda_list, 
        'dr_lambda':dr_lambda_list, 
        'switchdr_lambda':switchdr_lambda_list, 
        'dros_lambda':dros_lambda_list, 
        'sgipw_lambdas':sgipw_lambda_list, 
        'sgdr_lambdas':sgdr_lambda_list
        }

    ope_estimators = [
        IPWTuning(lambdas=ipw_lambda_list, tuning_method='slope', estimator_name='IPWTuning'), 
        DM(estimator_name='DM'), 
        DRTuning(lambdas=dr_lambda_list, tuning_method='slope',estimator_name='DRTuning'), 
        SNIPW(estimator_name='SNIPW'), 
        SwitchDRTuning(lambdas=switchdr_lambda_list, tuning_method='slope',estimator_name='SwitchDRTuning'), 
        SNDR(estimator_name='SNDR'),  
        DRosTuning(lambdas=dros_lambda_list, tuning_method='slope', estimator_name='DRosTuning'),
        SGIPWTuning(lambdas=sgipw_lambda_list, tuning_method='slope', estimator_name='SGIPWTuning'), 
        SGDRTuning(lambdas=sgdr_lambda_list, tuning_method='slope', estimator_name='SGDRTuning'), 
        ] # Do not change 'estimator_name' above

    if arguments.reward_type == 'binary':
        q_models = [RandomForestClassifier, lgb.LGBMClassifier, LogisticRegression]
    else:
        q_models = [RandomForestRegressor, lgb.LGBMRegressor, Ridge]


    pi_e = {}
    assert len(arguments.beta_list_for_pi_e) == len(arguments.model_list_for_pi_e), \
        'Must be len(arguments.beta_list_for_pi_e) == len(arguments.model_list_for_pi_e)'

    learned_model_dict = {}
    if os.path.dirname(__file__) == '':
        model_partial_path = './model/'
    else:
        model_partial_path = os.path.dirname(__file__) + '/model/'
    learned_model_dict['binary'] = [
        None, 
        model_partial_path+'ipw_lr_b.pickle', 
        model_partial_path+'ipw_rf_b.pickle', 
        model_partial_path+'qlr_lr_b.pickle', 
        model_partial_path+'qlr_rf_b.pickle'
        ]
    learned_model_dict['continuous'] = [
        None, 
        model_partial_path+'ipw_lr_c.pickle', 
        model_partial_path+'ipw_rf_c.pickle', 
        model_partial_path+'qlr_rr_c.pickle', 
        model_partial_path+'qlr_rf_c.pickle'
        ]

    for i_pi_e, model_num_for_pi_e in enumerate(arguments.model_list_for_pi_e):
        if model_num_for_pi_e != 0:
            with open(learned_model_dict[arguments.reward_type][model_num_for_pi_e], 'rb') as f:
                model_of_pi_e = pickle.load(f)

        if model_num_for_pi_e == 0:
            pi_e['beta_'+str(arguments.beta_list_for_pi_e[i_pi_e])] = ('beta', arguments.beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 1:
            pi_e['model_ipw_lr_beta_'+str(arguments.beta_list_for_pi_e[i_pi_e])] \
                = ('function', copy.deepcopy(model_of_pi_e), 1.0/arguments.beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 2:
            pi_e['model_ipw_rf_beta_'+str(arguments.beta_list_for_pi_e[i_pi_e])] \
                = ('function', copy.deepcopy(model_of_pi_e), 1.0/arguments.beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 3:
            if arguments.reward_type == 'binary':
                pi_e['model_qlr_lr_beta_'+str(arguments.beta_list_for_pi_e[i_pi_e])] \
                    = ('function', copy.deepcopy(model_of_pi_e), 1.0/arguments.beta_list_for_pi_e[i_pi_e])
            elif arguments.reward_type == 'continuous':
                pi_e['model_qlr_rr_beta_'+str(arguments.beta_list_for_pi_e[i_pi_e])] \
                    = ('function', copy.deepcopy(model_of_pi_e), 1.0/arguments.beta_list_for_pi_e[i_pi_e])
        elif model_num_for_pi_e == 4:
            pi_e['model_qlr_rf_beta_'+str(arguments.beta_list_for_pi_e[i_pi_e])] \
                = ('function', copy.deepcopy(model_of_pi_e), 1.0/arguments.beta_list_for_pi_e[i_pi_e])


    # set instance for evaluation of selection method
    evaluation_of_selection_method = EvaluationOfSelectionMethodWithSyntheticData(
        ope_estimators=ope_estimators, 
        q_models=q_models, 
        estimator_selection_metrics=arguments.metric, 
        n_actions=arguments.n_actions, 
        dim_context=arguments.dim_context, 
        beta_1=arguments.beta_1, 
        beta_2=arguments.beta_2, 
        reward_type=arguments.reward_type, 
        reward_function=reward_function_dict[arguments.reward_function], 
        n_rounds_1=arguments.n1, 
        n_rounds_2=arguments.n2, 
        test_ratio=None, 
        pi_e=pi_e, 
        n_data_generation=arguments.n_data_generation, 
        random_state=arguments.random_state
        )
    evaluation_of_selection_method.set_conventional_method_params(
        n_inner_bootstrap=arguments.n_bootstrap, 
        evaluation_data='partial_random'
        )
    if arguments.proposed_method == 'pasif':
        method_param_dict = {}
        for model_pi_e, beta, pasif_k, pasif_rw, pasif_bs, pasif_n_eopchs, pasif_opt, pasif_lr in \
            zip(arguments.model_list_for_pi_e, arguments.beta_list_for_pi_e, arguments.pasif_k, arguments.pasif_regularization_weight, arguments.pasif_batch_size, arguments.pasif_n_epochs, arguments.pasif_optimizer, arguments.pasif_lr) :
            if model_pi_e == 0:
                policy_name_for_key = 'beta_'+str(beta)
            elif model_pi_e == 1:
                policy_name_for_key = 'model_ipw_lr_beta_'+str(beta)
            elif model_pi_e == 2:
                policy_name_for_key = 'model_ipw_rf_beta_'+str(beta)
            elif model_pi_e == 3:
                if arguments.reward_type == 'binary':
                    policy_name_for_key = 'model_qlr_lr_beta_'+str(beta)
                elif arguments.reward_type == 'continuous':
                    policy_name_for_key = 'model_qlr_rr_beta_'+str(beta)
            elif model_pi_e == 4:
                policy_name_for_key = 'model_qlr_rf_beta_'+str(beta)
            method_param_dict[policy_name_for_key] = {
                'k':pasif_k, 
                'regularization_weight':pasif_rw, 
                'batch_size':pasif_bs, 
                'n_epochs':pasif_n_eopchs, 
                'optimizer':pasif_optimizer_dict[pasif_opt], 
                'lr':pasif_lr
                }
        evaluation_of_selection_method.set_proposed_method_params(
            method_name=arguments.proposed_method, 
            method_params=method_param_dict, 
            n_inner_bootstrap=arguments.n_bootstrap
            )
    elif arguments.proposed_method == 'pass':
        method_param_dict = {'k':arguments.pass_k, 'alpha':arguments.pass_alpha, 'tuning':False}
        evaluation_of_selection_method.set_proposed_method_params(
            method_name=arguments.proposed_method, 
            method_params=method_param_dict, 
            n_inner_bootstrap=arguments.n_bootstrap
            )
    elif arguments.proposed_method == 'pass_tuning':
        method_param_dict = {'k':arguments.pass_tuning_k, 'alpha':None, 'tuning':True}
        evaluation_of_selection_method.set_proposed_method_params(
            method_name='pass', 
            method_params=method_param_dict, 
            n_inner_bootstrap=arguments.n_bootstrap
            )

    processing_time_list.append(time.time())
    print('set ground truth', time.gmtime())
    evaluation_of_selection_method.set_ground_truth(n_sampling=arguments.gt_n_sampling)


    # evaluation of conventional method
    processing_time_list.append(time.time())
    print('evaluation of conventional selection method', time.gmtime())
    evaluation_of_selection_method.evaluate_conventional_selection_method()
    
    # evaluation of proposed estimator selection method
    processing_time_list.append(time.time())
    print('evaluation of proposed selection method', time.gmtime())
    evaluation_of_selection_method.evaluate_proposed_selection_method()

    # logging (output)
    processing_time_list.append(time.time())
    print('save result')
    # create folder for saving log
    log_dir_path = arguments.save_dir + '/log'
    if not os.path.exists(log_dir_path):
        os.mkdir(log_dir_path)
    log_dir_path = log_dir_path + '/beta_1_' + str(arguments.beta_1) + '_beta_2_' + str(arguments.beta_2)
    log_dir_path = mk_log_dir(dir_name=log_dir_path)
    # config
    save_config(dir_name=log_dir_path, args=arguments, ope_estimators=ope_estimators, hypara_dict=hypara_dict, q_models=q_models)
    # all result
    with open(log_dir_path+'/evaluation_of_selection_method.pickle', 'wb') as f:
        pickle.dump(evaluation_of_selection_method, f)
    # summarized result
    save_summarized_results(dir_name=log_dir_path, evaluation_of_selection_method=evaluation_of_selection_method)
    # figure
    save_figure(dir_name=log_dir_path, metric_name='relative_regret', pi_e=pi_e, evaluation_of_selection_method=evaluation_of_selection_method)
    save_figure(dir_name=log_dir_path, metric_name='rank_correlation_coefficient', pi_e=pi_e, evaluation_of_selection_method=evaluation_of_selection_method)
    # processing time
    processing_time_list.append(time.time())
    file_path = log_dir_path + '/processing_time.csv'
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['process', 'hour', 'minute', 'second'])
        processtime = convert_sec_to_hms(sec=processing_time_list[1]-processing_time_list[0])
        writer.writerow(['set_ground_truth', processtime[0], processtime[1], processtime[2]])
        processtime = convert_sec_to_hms(sec=processing_time_list[2]-processing_time_list[1])
        writer.writerow(['eval_of_conventional_method', processtime[0], processtime[1], processtime[2]])
        processtime = convert_sec_to_hms(sec=processing_time_list[3]-processing_time_list[2])
        writer.writerow(['eval_of_proposed_method', processtime[0], processtime[1], processtime[2]])
        processtime = convert_sec_to_hms(sec=processing_time_list[4]-processing_time_list[3])
        writer.writerow(['logging', processtime[0], processtime[1], processtime[2]])
        processtime = convert_sec_to_hms(sec=processing_time_list[4]-processing_time_list[0])
        writer.writerow(['total', processtime[0], processtime[1], processtime[2]])


