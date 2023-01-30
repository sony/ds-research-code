import imp
from posixpath import split
import numpy as np
import pandas as pd
import random
import copy
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim


class DataSplittingByPasif:
    """split data by pasif
    """
    def __init__(self, batch_bandit_feedback, action_dist_by_pi_e, random_state=None):
        """set basic info

        Args:
            batch_bandit_feedback (dict): batch bandit feedback
            action_dist_by_pi_e (np.array): action dist by evalation policy
            random_state (int, optional): Defaults to None.
        """
        self.batch_bandit_feedback = batch_bandit_feedback
        self.action_dist_by_pi_e = action_dist_by_pi_e
        if random_state is None:
            self.random_state = random.randint(1,10000000)
        else:
            self.random_state = random_state

    def set_params(
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
        self.k = k
        self.regularization_weight = regularization_weight
        if batch_size is None:
            self.batch_size = self.batch_bandit_feedback['n_rounds']
        else:
            self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.lr = lr

        # pscore by evaluation policy
        self.pscore_by_pi_e = np.zeros_like(self.batch_bandit_feedback['pscore'])
        for action in range(self.batch_bandit_feedback['n_actions']):
            self.pscore_by_pi_e[self.batch_bandit_feedback['action']==action] = self.action_dist_by_pi_e[self.batch_bandit_feedback['action']==action, action, 0]


    def train_importance_fitting(self):
        """train NN by importance fitting
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device_type', device)

        # convert data to Tensor type
        x_and_a = np.concatenate([self.batch_bandit_feedback['context'], self.batch_bandit_feedback['action'].reshape(self.batch_bandit_feedback['action'].shape[0],1)], axis=-1)
        x_and_a = torch.from_numpy(x_and_a).float() # data of actext and action
        w = self.pscore_by_pi_e / self.batch_bandit_feedback['pscore']
        w = torch.from_numpy(w).float() # ideal importance weight (w)

        pi_b = torch.from_numpy(self.batch_bandit_feedback['pi_b']).float() # behavior policy

        pscore = torch.from_numpy(self.batch_bandit_feedback['pscore']).float() # propensity score

        # create tensor data set (context_and_actual_action, importance_weight, behavior_policy, propensity_score)
        tensor_dataset = torch.utils.data.TensorDataset(x_and_a, w, pi_b, pscore)

        # create data loader
        dataloader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        # some values needed to calculate loss
        n_actions = self.batch_bandit_feedback['n_actions']
        k = self.k
        regularization_weight = self.regularization_weight


        class CustomLoss(nn.Module):
            """loss for importance fitting
            """
            def __init__(self):
                super().__init__()

            def forward(self, outputs, targets):
                # outputs: (batch_size, n_actions+1), outputs with actual action and fixed actions
                # targets: (batch_size, n_actions+2), w, pi_b, pscore
                for action in range(1, 1+n_actions):
                    if action == 1:
                        marginal_p = targets[:,action]*outputs[:,action] # marginal_p = \Sigma_{a} \pi_b(a|x) * \rho_{\theta}(a|x)
                    else:
                        marginal_p += targets[:,action]*outputs[:,action]
                tilde_p_e = targets[:,-1] * (outputs[:,0] / marginal_p)
                tilde_p_b = targets[:,-1] * ((1.0-outputs[:,0]) / (1.0-marginal_p))
                tilde_w = tilde_p_e / tilde_p_b # importance weight by NN
                self.loss_d = ((tilde_w - targets[:,0])**2).mean()
                self.loss_r = ((marginal_p - k)**2).mean()
                loss = self.loss_d + regularization_weight * self.loss_r
                return loss

        criterion = CustomLoss()

        # dim of context+action
        dim_context = self.batch_bandit_feedback['context'].shape[1] + 1

        class Net(nn.Module):
            """NN for \rho
            """
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(dim_context, 100)
                self.fc2 = nn.Linear(100, 100)
                self.fc3 = nn.Linear(100, 1)
            
            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.relu(x)
                x = self.fc3(x)
                # x = F.sigmoid(x)
                x = torch.sigmoid(x) # it is recommended to use torch.sigmoid rather than F.sigmoid
                return x
                    
        torch.manual_seed(self.random_state)
        self.net = Net().to(device)

        optimizer = self.optimizer(self.net.parameters(), lr=self.lr)

        print('train start; regularization_weight=', regularization_weight)
        for epoch in range(self.n_epochs):
            if (epoch+1)%50 == 0 or epoch ==0:
                print('Epoch {}/{}'.format(epoch+1, self.n_epochs))
                print('-------------')
            
            # for phase in ['train', 'val']: # To cut the calculation time, we only conduct train phase (There is no problem because all the data is used for training)
            for phase in ['train']:
                
                if phase == 'train':
                    # Set model to training mode
                    self.net.train()
                else:
                    # Set model to evaluation mode
                    self.net.eval()
                
                # sum of loss
                epoch_loss_d = 0.0
                epoch_loss_r = 0.0
                epoch_loss = 0.0
                
                # Extract data from dataloader in batch
                for t_x_and_a, t_w, t_pi_b, t_p in dataloader:

                    t_x_and_a, t_w, t_pi_b, t_p = t_x_and_a.to(device), t_w.to(device), t_pi_b.to(device), t_p.to(device)

                    # initialize optimizer
                    optimizer.zero_grad()
                    
                    # Set to calculate the gradient only during learning
                    with torch.set_grad_enabled(phase=='train'):

                        output_actual = self.net(t_x_and_a)
                        output_list = [output_actual]
                        for action in range(self.batch_bandit_feedback['n_actions']):
                            t_x_and_fixed_action = copy.deepcopy(t_x_and_a)
                            t_x_and_fixed_action[:,-1] = action
                            output_fixed_action = self.net(t_x_and_fixed_action)
                            output_list.append(output_fixed_action)
                        outputs = torch.cat(output_list, axis=1)
                        
                        t_p = t_p.reshape((t_p.shape[0],1))
                        t_w = t_w.reshape((t_w.shape[0],1))                
                        t_pi_b = t_pi_b.reshape((t_pi_b.shape[0], t_pi_b.shape[1]))
                        targets = torch.cat([t_w, t_pi_b, t_p], axis=1)
                        
                        # calc loss
                        loss = criterion(outputs=outputs, targets=targets)
                        
                        # Backpropagation during training
                        if phase == 'train':
                            # Backpropagation calculation
                            loss.backward()
                            # Parameter update
                            optimizer.step()
                        
                        # Calculation of iteration results
                        # Update total loss
                        epoch_loss += loss.item() * t_x_and_a.shape[0]
                        epoch_loss_d += criterion.loss_d * t_x_and_a.shape[0]
                        epoch_loss_r += criterion.loss_r * t_x_and_a.shape[0]


                # Display loss for each epoch
                epoch_loss = epoch_loss / len(dataloader.dataset)
                epoch_loss_d = epoch_loss_d / len(dataloader.dataset)
                epoch_loss_r = epoch_loss_r / len(dataloader.dataset)
                if (epoch+1)%50 == 0 or epoch ==0:
                    print('{} TotalLoss: {:.10f} LossD: {:.10f} LossR: {:.10f}'.format(phase, epoch_loss, epoch_loss_d, epoch_loss_r))

        print('train end')
        self.final_loss = float(epoch_loss)
        self.final_loss_d = float(epoch_loss_d)
        self.final_loss_r = float(epoch_loss_r)

        # dict to have some info for spliting the dataset
        self.split_info_dict = {}

        # sampling probability
        sampling_probability = self.net(x_and_a.to(device))
        sampling_probability = sampling_probability.to(torch.device('cpu'))
        sampling_probability = sampling_probability.detach().numpy().copy()
        sampling_probability = sampling_probability.reshape((sampling_probability.shape[0]))
        self.split_info_dict['sampling_probability'] = sampling_probability


        if np.isnan(self.split_info_dict['sampling_probability']).sum() > 0:
            # case that NN training does not go well and nan is output
            self.split_info_dict['rand_num'] = None
            self.split_info_dict['sampling_indicator'] = None
            self.split_info_dict['marginal_p'] = None
            self.split_info_dict['pi_b1'] = None
            self.split_info_dict['pi_b1_pscore'] = None
            self.split_info_dict['pi_b2'] = None
            self.split_info_dict['pi_b2_pscore'] = None
            self.batch_feedback_1 = None
            self.cf_action_dist_1 = None
            self.batch_feedback_2 = None
            self.cf_action_dist_2 = None


        else:    
            # random number for sampling
            np.random.seed(seed=self.random_state)
            rand_num = np.random.rand(self.batch_bandit_feedback['n_rounds'])
            self.split_info_dict['rand_num'] = rand_num

            # sampling indicator
            sampling_indicator = np.zeros((self.batch_bandit_feedback['n_rounds']), dtype=int)
            sampling_indicator[sampling_probability > rand_num] = 1
            self.split_info_dict['sampling_indicator'] = sampling_indicator


            # marginal p
            for action in range(self.batch_bandit_feedback['n_actions']):
                x_and_fixed_action = copy.deepcopy(x_and_a).to(device)
                x_and_fixed_action[:,-1] = action
                t_output_fixed_action = self.net(x_and_fixed_action)
                t_output_fixed_action = t_output_fixed_action.to(torch.device('cpu'))
                t_output_fixed_action = t_output_fixed_action.detach().numpy().copy()
                if action == 0:
                    output_fixed_action = t_output_fixed_action
                    marginal_p = self.batch_bandit_feedback['pi_b'][:, action, 0] * t_output_fixed_action[:,0]
                else:
                    output_fixed_action = np.concatenate([output_fixed_action, t_output_fixed_action],axis=-1)
                    marginal_p += self.batch_bandit_feedback['pi_b'][:, action, 0] * t_output_fixed_action[:,0]
            self.split_info_dict['marginal_p'] = marginal_p

            # action dist by pi_b1 (quasi pi_e)
            self.split_info_dict['pi_b1'] = self.batch_bandit_feedback['pi_b'].copy()
            for action in range(self.batch_bandit_feedback['n_actions']):
                self.split_info_dict['pi_b1'][:,action,0] = self.batch_bandit_feedback['pi_b'][:,action,0] * (output_fixed_action[:,action] / self.split_info_dict['marginal_p'])

            # pscore by pi_b1 (quasi pi_e)
            self.split_info_dict['pi_b1_pscore'] = np.zeros_like(self.batch_bandit_feedback['pscore'])
            for action in range(self.batch_bandit_feedback['n_actions']):
                self.split_info_dict['pi_b1_pscore'][self.batch_bandit_feedback['action']==action] \
                    = self.split_info_dict['pi_b1'][self.batch_bandit_feedback['action']==action, action, 0].copy()


            # action dist by pi_b2 (quasi pi_b)
            self.split_info_dict['pi_b2'] = self.batch_bandit_feedback['pi_b'].copy()
            for action in range(self.batch_bandit_feedback['n_actions']):
                self.split_info_dict['pi_b2'][:,action,0] = self.batch_bandit_feedback['pi_b'][:,action,0] * ((1.0-output_fixed_action[:,action]) / (1.0-self.split_info_dict['marginal_p']))

            # pscore by pi_b2 (quasi pi_b)
            self.split_info_dict['pi_b2_pscore'] = np.zeros_like(self.batch_bandit_feedback['pscore'])
            for action in range(self.batch_bandit_feedback['n_actions']):
                self.split_info_dict['pi_b2_pscore'][self.batch_bandit_feedback['action']==action] \
                    = self.split_info_dict['pi_b2'][self.batch_bandit_feedback['action']==action, action, 0].copy()


            if sampling_indicator.sum() < 1:
                self.batch_feedback_1 = None
                self.cf_action_dist_1 = None
                self.batch_feedback_2 = None
                self.cf_action_dist_2 = None

            else:
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
                            self.batch_feedback_1[feedback_key] = self.batch_bandit_feedback[feedback_key]
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


