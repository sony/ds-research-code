# Copyright (c) 2023 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.

import numpy as np
import pandas as pd
import random



def calculate_alpha_prime_min_and_max(i_pi_b, i_pi_e, k, n_actions):
    """return alpha_prime_max/min for 1 sample (i)

    Args:
        i_pi_b (np.array): action dist for sample i by pi_b
        i_pi_e (np.array): action dist for sample i by pi_e
        k (float): param k
        n_actions (int): Number of actions

    Returns:
        float, float: alpha_prime_min and  alpha_prime_max
    """
    alpha_prime_max = 1
    alpha_prime_min = -1
    
    for i_action in range(n_actions):
        if i_pi_e[i_action]-i_pi_b[i_action]==0:
            pass
        else:
            lower_limit_pi_b1 = max(0, 1-k+k*i_pi_b[i_action])
            upper_limit_pi_b1 = min(1, k*i_pi_b[i_action])
            
            temp_alpha_prime_max = max((upper_limit_pi_b1-i_pi_b[i_action])/(i_pi_e[i_action]-i_pi_b[i_action]), 
                                    (lower_limit_pi_b1-i_pi_b[i_action])/(i_pi_e[i_action]-i_pi_b[i_action]))

            temp_alpha_prime_min = min((upper_limit_pi_b1-i_pi_b[i_action])/(i_pi_e[i_action]-i_pi_b[i_action]), 
                                    (lower_limit_pi_b1-i_pi_b[i_action])/(i_pi_e[i_action]-i_pi_b[i_action]))
        
            if temp_alpha_prime_max < alpha_prime_max:
                alpha_prime_max = temp_alpha_prime_max
            if temp_alpha_prime_min > alpha_prime_min:
                alpha_prime_min = temp_alpha_prime_min

    # 少数の誤差で万が一alpha_prime_max < 0、alpha_prime_min > 0になったケースの対処
    # Dealing with cases where alpha_prime_max < 0 or alpha_prime_min > 0 by any chance with a decimal error
    if alpha_prime_max < 0:
        alpha_prime_max = 0
    if alpha_prime_min > 0:
        alpha_prime_min = 0
        
    return alpha_prime_min, alpha_prime_max




class DataSplittingByPass:
    """split data by pass
    """

    def __init__(self, batch_bandit_feedback, action_dist_by_pi_e, k=2.0, alpha=1.0, random_state=None):
        """set basic info

        Args:
            batch_bandit_feedback (dict): batch bandit feedback
            action_dist_by_pi_e (np.array): action dist by evalation policy
            k (float, optional): Defaults to 2.0. param of pass.
            alpha (float, optional): Defaults to 1.0. param of pass.
            random_state (int, optional):  Defaults to None.
        """
        self.batch_bandit_feedback = batch_bandit_feedback
        self.action_dist_by_pi_e = action_dist_by_pi_e
        self.k = k
        self.k_inverse = 1/k
        self.alpha = alpha
        if random_state is None:
            self.random_state = random.randint(1,10000000)
        else:
            self.random_state = random_state


    def split_data(self):
        """split data
        """
        # dict to have some info for spliting the dataset
        self.split_info_dict = {}

        self.split_info_dict['pi_b']  = self.batch_bandit_feedback['pi_b'].copy()
        self.split_info_dict['pi_e']  = self.action_dist_by_pi_e.copy()

        # pi_e p_score
        pi_e_pscore = np.zeros_like(self.batch_bandit_feedback['pscore'])
        for i_action in range(self.batch_bandit_feedback['n_actions']):
            pi_e_pscore[self.batch_bandit_feedback['action']==i_action] = self.split_info_dict['pi_e'][self.batch_bandit_feedback['action']==i_action, i_action, 0]
        self.split_info_dict['pi_e_pscore'] = pi_e_pscore



        # min value of pi_b1
        min_value_of_pi_b1 = 1 - self.k + self.k * self.split_info_dict['pi_b']
        min_value_of_pi_b1[min_value_of_pi_b1<0] = 0
        self.split_info_dict['min_value_of_pi_b1'] = min_value_of_pi_b1

        # max value of pi_b1
        max_value_of_pi_b1 = self.k * self.split_info_dict['pi_b']
        max_value_of_pi_b1[max_value_of_pi_b1>1] = 1
        self.split_info_dict['max_value_of_pi_b1'] = max_value_of_pi_b1

        # pi_b1
        pi_b1 = np.zeros_like(self.split_info_dict['pi_b'])
        for i_sample in range(self.batch_bandit_feedback['n_rounds']):
            i_pi_b= self.split_info_dict['pi_b'][i_sample, :, 0]
            i_pi_e = self.split_info_dict['pi_e'][i_sample, :, 0]
            alpha_prime_min, alpha_prime_max = \
                calculate_alpha_prime_min_and_max(i_pi_b=i_pi_b, 
                                        i_pi_e=i_pi_e, 
                                        k=self.k, 
                                        n_actions=self.batch_bandit_feedback['n_actions'])
            if self.alpha > 0:
                alpha_prime = self.alpha * alpha_prime_max
            elif self.alpha <= 0:
                alpha_prime = (-1) * self.alpha * alpha_prime_min
            
            pi_b1[i_sample, :, 0] = (1.0 - alpha_prime) * i_pi_b + alpha_prime * i_pi_e

        self.split_info_dict['pi_b1'] = pi_b1

        # pi_b1 p_score
        pi_b1_pscore = np.zeros_like(self.batch_bandit_feedback['pscore'])
        for i_action in range(self.batch_bandit_feedback['n_actions']):
            pi_b1_pscore[self.batch_bandit_feedback['action']==i_action] = self.split_info_dict['pi_b1'][self.batch_bandit_feedback['action']==i_action, i_action, 0]
        self.split_info_dict['pi_b1_pscore'] = pi_b1_pscore

        # pi_b2
        pi_b2 = (self.k/(self.k-1)) * (self.split_info_dict['pi_b'] - self.k_inverse * self.split_info_dict['pi_b1'])
        self.split_info_dict['pi_b2'] = pi_b2

        # pi_b2 p_score
        pi_b2_pscore = np.zeros_like(self.batch_bandit_feedback['pscore'])
        for i_action in range(self.batch_bandit_feedback['n_actions']):
            pi_b2_pscore[self.batch_bandit_feedback['action']==i_action] = self.split_info_dict['pi_b2'][self.batch_bandit_feedback['action']==i_action, i_action, 0]
        self.split_info_dict['pi_b2_pscore'] = pi_b2_pscore


        # p(g=1|x_i,a_i), probability from D to D_1
        sampling_probability = ( self.k_inverse * self.split_info_dict['pi_b1_pscore'] ) / self.batch_bandit_feedback['pscore']
        self.split_info_dict['sampling_probability'] = sampling_probability
        
        # random number for sampling
        np.random.seed(seed=self.random_state)
        rand_num = np.random.rand(self.batch_bandit_feedback['n_rounds'])
        self.split_info_dict['rand_num'] = rand_num

        # sampling indicator (1:from D to D_1, 0:remain and become D_2)
        sampling_indicator = np.zeros((self.batch_bandit_feedback['n_rounds']), dtype=int)
        sampling_indicator[sampling_probability > rand_num] = 1
        self.split_info_dict['sampling_indicator'] = sampling_indicator
        
        # create new dataset
        self.batch_feedback_1 = {}
        for feedback_key in self.batch_bandit_feedback.keys():
            if feedback_key == 'n_rounds':
                self.batch_feedback_1[feedback_key] = sum(self.split_info_dict['sampling_indicator'])
            elif feedback_key == 'n_actions':
                self.batch_feedback_1[feedback_key] = self.batch_bandit_feedback[feedback_key]
            elif feedback_key == 'context':
                self.batch_feedback_1[feedback_key] = self.batch_bandit_feedback[feedback_key][self.split_info_dict['sampling_indicator']==1, :].copy()
            elif feedback_key == 'action_context':
                if self.batch_bandit_feedback[feedback_key] is None:
                    self.batch_feedback_1[feedback_key] = None
                else:
                    self.batch_feedback_1[feedback_key] = self.batch_bandit_feedback[feedback_key].copy()
            elif feedback_key == 'action':
                self.batch_feedback_1[feedback_key] = self.batch_bandit_feedback[feedback_key][self.split_info_dict['sampling_indicator']==1].copy()
            elif feedback_key == 'position':
                if self.batch_bandit_feedback[feedback_key] is None:
                    self.batch_feedback_1[feedback_key] = None
                else:
                    self.batch_feedback_1[feedback_key] = self.batch_bandit_feedback[feedback_key][self.split_info_dict['sampling_indicator']==1].copy()
            elif feedback_key == 'reward':
                self.batch_feedback_1[feedback_key] = self.batch_bandit_feedback[feedback_key][self.split_info_dict['sampling_indicator']==1].copy()
            elif feedback_key == 'expected_reward':
                if self.batch_bandit_feedback[feedback_key] is None:
                    self.batch_feedback_1[feedback_key] = None
                else:
                    self.batch_feedback_1[feedback_key] = self.batch_bandit_feedback[feedback_key][self.split_info_dict['sampling_indicator']==1, :].copy()
            elif feedback_key == 'pi_b':
                if self.batch_bandit_feedback[feedback_key] is None:
                    self.batch_feedback_1[feedback_key] = None
                else:
                    self.batch_feedback_1[feedback_key] = self.split_info_dict['pi_b1'][self.split_info_dict['sampling_indicator']==1, :, :].copy()
            elif feedback_key == 'pscore':
                self.batch_feedback_1[feedback_key] = self.split_info_dict['pi_b1_pscore'][self.split_info_dict['sampling_indicator']==1].copy()


        self.batch_feedback_2 = {}
        for feedback_key in self.batch_bandit_feedback.keys():
            if feedback_key == 'n_rounds':
                self.batch_feedback_2[feedback_key] = self.batch_bandit_feedback['n_rounds'] - sum(self.split_info_dict['sampling_indicator'])
            elif feedback_key == 'n_actions':
                self.batch_feedback_2[feedback_key] = self.batch_bandit_feedback[feedback_key]
            elif feedback_key == 'context':
                self.batch_feedback_2[feedback_key] = self.batch_bandit_feedback[feedback_key][self.split_info_dict['sampling_indicator']==0, :].copy()
            elif feedback_key == 'action_context':
                if self.batch_bandit_feedback[feedback_key] is None:
                    self.batch_feedback_2[feedback_key] = None
                else:
                    self.batch_feedback_2[feedback_key] = self.batch_bandit_feedback[feedback_key].copy()
            elif feedback_key == 'action':
                self.batch_feedback_2[feedback_key] = self.batch_bandit_feedback[feedback_key][self.split_info_dict['sampling_indicator']==0].copy()
            elif feedback_key == 'position':
                if self.batch_bandit_feedback[feedback_key] is None:
                    self.batch_feedback_2[feedback_key] = None
                else:
                    self.batch_feedback_2[feedback_key] = self.batch_bandit_feedback[feedback_key][self.split_info_dict['sampling_indicator']==0].copy()
            elif feedback_key == 'reward':
                self.batch_feedback_2[feedback_key] = self.batch_bandit_feedback[feedback_key][self.split_info_dict['sampling_indicator']==0].copy()
            elif feedback_key == 'expected_reward':
                if self.batch_bandit_feedback[feedback_key] is None:
                    self.batch_feedback_2[feedback_key] = None
                else:
                    self.batch_feedback_2[feedback_key] = self.batch_bandit_feedback[feedback_key][self.split_info_dict['sampling_indicator']==0, :].copy()
            elif feedback_key == 'pi_b':
                if self.batch_bandit_feedback[feedback_key] is None:
                    self.batch_feedback_2[feedback_key] = None
                else:
                    self.batch_feedback_2[feedback_key] = self.split_info_dict['pi_b2'][self.split_info_dict['sampling_indicator']==0, :, :].copy()
            elif feedback_key == 'pscore':
                self.batch_feedback_2[feedback_key] = self.split_info_dict['pi_b2_pscore'][self.split_info_dict['sampling_indicator']==0].copy()

        # create counterfactual action dist
        self.cf_action_dist_1 = self.split_info_dict['pi_b2'][self.split_info_dict['sampling_indicator']==1, :, :]
        self.cf_action_dist_2 = self.split_info_dict['pi_b1'][self.split_info_dict['sampling_indicator']==0, :, :]


    def get_split_data(self):
        """get data after splitting

        Returns:
            dict, dict, np.array, np.array: batch1, counterfacal action dist1, batch2, counterfacal action dist2
        """
        return self.batch_feedback_1, self.cf_action_dist_1, self.batch_feedback_2, self.cf_action_dist_2


    def get_importance_weight(self):
        """get importance weight for all sample

        Returns:
            np.array: array of importance weight
        """
        return self.split_info_dict['pi_b1_pscore']/self.split_info_dict['pi_b2_pscore']


    def get_mse_of_importance_weight(self):
        """get mse of importance weight

        Returns:
            float: mse of ideal and actual importance weight
        """
        ideal_importance_weight = self.split_info_dict['pi_e_pscore']/self.batch_bandit_feedback['pscore']
        actual_importance_weight =  self.split_info_dict['pi_b1_pscore']/self.split_info_dict['pi_b2_pscore']
        mse = ((ideal_importance_weight - actual_importance_weight)**2).mean()
        return mse


