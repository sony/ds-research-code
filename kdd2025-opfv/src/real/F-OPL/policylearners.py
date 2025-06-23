# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import datetime
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import conf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from estimators_time import fourier_scalar, fourier_vec
from sklearn.utils import check_random_state
from tqdm import tqdm
from utils import (
    BehaviorPolicyDataset,
    GradientBasedPolicyDataset,
    OPFVDataset,
    Prognosticatordataset,
    RegBasedPolicyDataset,
    softmax,
)

DAYS_IN_A_YEAR = 365
DAYS_IN_A_WEEK = 7

NUM_UNIQUE_DAY_OF_WEEK = 7
NUM_UNIQUE_HOURS = 24
NUM_UNIQUE_DAY_OFF_OR_NOR = 2


def convert_unix_time_to_num_time_structures_after_the_oldest_time(
    unix_time, t_oldest, num_time_structures_in_a_week
):
    unix_time_datetime = datetime.datetime.utcfromtimestamp(unix_time)
    t_oldest_datetime = datetime.datetime.utcfromtimestamp(t_oldest)

    elapsed_time = unix_time_datetime - t_oldest_datetime

    elapsed_days = elapsed_time.days

    num_time_structures_after_the_oldest_time = int(
        np.ceil(elapsed_days / DAYS_IN_A_WEEK * num_time_structures_in_a_week)
    )

    return num_time_structures_after_the_oldest_time


@dataclass
class RegBasedPolicyLearner:
    dim_x: int  # d_x
    num_actions: int  # |A|
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 32
    evaluation_rate_init: float = 0.01
    alpha: float = 1e-6  # weight decay used in optimizer
    log_eps: float = 1e-10  # this variable is not used in RegBasedPolicyLearner
    solver: str = "adagrad"
    max_iter: int = 50  # number of epochs
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x

        # Set the activation layer (Tanh, ReLU, or ELU)
        # Default ELU
        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        # Define the model
        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)

        # the length of the vector is the number of epochs
        self.train_loss = []
        self.train_value = []
        self.test_value = []

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, a, r = dataset["context"], dataset["action"], dataset["reward"]

        # Instatiate the solver (adagrad or adam)
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.evaluation_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.evaluation_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        # Create the training data loader
        training_data_loader = self._create_train_data_for_opl(x, a, r)

        # start policy training
        q_x_a_train, q_x_a_test = (
            dataset["expected_reward"],
            dataset_test["expected_reward"],
        )
        for _ in range(self.max_iter):
            if _ == 0:
                self.nn_model.eval()
                # Calcualte the trained policy using the training data
                pi_train = self.predict(dataset)
                # Calcualte the true value of the trained policy
                self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

                # Calcualte the trained policy using the test data
                pi_test = self.predict(dataset_test)
                # Calcualte the true value of the test policy
                self.test_value.append((q_x_a_test * pi_test).sum(1).mean())
            loss_epoch = 0.0
            # Start training mode
            self.nn_model.train()
            for x, a, r in training_data_loader:
                # Set the gradient to zero
                optimizer.zero_grad()
                # Calculate the estimated expected reward (batch size * |A|)
                q_hat = self.nn_model(x)
                # idx = (0, 1, \cdots, batch size)
                idx = torch.arange(a.shape[0], dtype=torch.long)
                # Calculate the loss
                loss = ((r - q_hat[idx, a]) ** 2).mean()
                # Calculate the gradient using backward proopagation
                loss.backward()
                # Update the parameters using the calculated gradient
                optimizer.step()
                loss_epoch += loss.item()
            # Calcualte the trained policy using the training data
            pi_train = self.predict(dataset)
            # Calcualte the true value of the trained policy
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

            # Calcualte the trained policy using the test data
            pi_test = self.predict(dataset_test)
            # Calcualte the true value of the test policy
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

            self.train_loss.append(loss_epoch)

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
    ) -> tuple:
        dataset = RegBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def predict(self, dataset_test: np.ndarray, beta: float = 10) -> np.ndarray:
        # Set the evaluation mode
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float()
        q_hat = self.nn_model(x).detach().numpy()

        return softmax(beta * q_hat)

    # This method is not used in RegBasedPolicyLearner
    def predict_q(self, dataset_test: np.ndarray) -> np.ndarray:
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float()

        return self.nn_model(x).detach().numpy()


@dataclass
class GradientBasedPolicyLearner:
    dim_x: int  # d_x
    num_actions: int  # |A|
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 32
    evaluation_rate_init: float = 0.01
    alpha: float = 1e-6  # weight decay used in optimizer
    imit_reg: float = 0.0  #### this is the different param from RegBased
    log_eps: float = 1e-10  # to calculate the log the input of the log should not be zero so engineering trick
    solver: str = "adagrad"
    max_iter: int = 50  # number of epochs
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x

        # Set the activation layer (Tanh, ReLU, or ELU)
        # Default ELU
        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        # Define the model
        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(
            ("softmax", nn.Softmax(dim=1))
        )  ######## here is the difference from regression based policy learner ########

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)

        # the length of the vector is the number of epochs
        self.train_loss = []
        self.train_value = []
        self.test_value = []

    def fit(self, dataset: dict, dataset_test: dict, q_hat: np.ndarray = None) -> None:
        x, a, r = dataset["context"], dataset["action"], dataset["reward"]
        pscore, pi_0 = (
            dataset["pscore"],
            np.squeeze(dataset["pi_b"]),
        )  ####### this is also the difference from Regression based Policy Learner ######
        # if \hat{q}(x, a) is not provided, then create zero vector
        if q_hat is None:
            q_hat = np.zeros((r.shape[0], self.num_actions))

        # Instatiate the solver (adagrad or adam)
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.evaluation_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.evaluation_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        # Create the training data loader
        training_data_loader = self._create_train_data_for_opl(
            x,
            a,
            r,
            pscore,
            q_hat,
            pi_0,
        )

        # start policy training
        q_x_a_train, q_x_a_test = (
            dataset["expected_reward"],
            dataset_test["expected_reward"],
        )
        for _ in range(self.max_iter):
            if _ == 0:
                self.nn_model.eval()
                # Calcualte the trained policy using the training data
                pi_train = self.predict(dataset)
                # Calcualte the true value of the trained policy
                self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

                # Calcualte the trained policy using the test data
                pi_test = self.predict(dataset_test)
                # Calcualte the true value of the test policy
                self.test_value.append((q_x_a_test * pi_test).sum(1).mean())
            loss_epoch = 0.0
            # Start training mode
            self.nn_model.train()
            for x, a, r, p, q_hat_, pi_0_ in training_data_loader:
                # Set the gradient to zero
                optimizer.zero_grad()
                # Calculate the estimated trained policy at time _ (batch size * |A|)
                pi = self.nn_model(x)
                # Calculate the loss
                loss = -self._estimate_policy_gradient(
                    a=a,
                    r=r,
                    pscore=p,
                    q_hat=q_hat_,
                    pi_0=pi_0_,
                    pi=pi,
                ).mean()
                # Calculate the gradient using backward proopagation
                loss.backward()
                # Update the parameters using the calculated gradient
                optimizer.step()
                loss_epoch += loss.item()

            # Calcualte the trained policy using the training data
            pi_train = self.predict(dataset)
            # Calcualte the true value of the trained policy
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

            # Calcualte the trained policy using the test data
            pi_test = self.predict(dataset_test)
            # Calcualte the true value of the test policy
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

            self.train_loss.append(loss_epoch)

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        pi_0: np.ndarray,
    ) -> tuple:
        dataset = GradientBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        q_hat: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)

        q_hat_factual = q_hat[idx, a]
        iw = current_pi[idx, a] / pscore
        estimated_policy_grad_arr = iw * (r - q_hat_factual) * log_prob[idx, a]
        estimated_policy_grad_arr += torch.sum(q_hat * current_pi * log_prob, dim=1)

        # imitation regularization
        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, a]

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float()
        return self.nn_model(x).detach().numpy()


@dataclass
class Prognosticator:
    dim_x: int  # d_x
    num_actions: int  # |A|
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 32
    evaluation_rate_init: float = 0.01
    alpha: float = 1e-6  # weight decay used in optimizer
    imit_reg: float = 0.0
    log_eps: float = 1e-10  # to calculate the log the input of the log should not be zero so engineering trick
    solver: str = "adagrad"
    max_iter: int = 50  # number of epochs
    random_state: int = 12345
    num_features_for_Prognosticator: int = conf.num_time_structure_for_logged_data
    num_time_structures_in_a_week: int = conf.num_time_structures_in_a_week
    time_structure_for_episode_division: callable = (
        conf.phi_scalar_func_for_Prognosticator
    )
    time_feature_func_for_Prognosticator_scalar: callable = fourier_scalar
    time_feature_func_for_Prognosticator_vec: callable = fourier_vec
    num_parameters: int = 5
    t_oldest: int = None
    t_now: int = None
    t_future: int = None
    time_test: np.ndarray = None
    num_time_learn: int = (conf.num_time_learn,)

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x

        # Set the activation layer (Tanh, ReLU, or ELU)
        # Default ELU
        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        max_num_time_structures_after_t_oldest = (
            convert_unix_time_to_num_time_structures_after_the_oldest_time(
                unix_time=self.t_future,
                t_oldest=self.t_oldest,
                num_time_structures_in_a_week=self.num_time_structures_in_a_week,
            )
        )
        self.max_num_time_structures_after_t_oldest = int(
            np.ceil(max_num_time_structures_after_t_oldest)
        )
        K = convert_unix_time_to_num_time_structures_after_the_oldest_time(
            unix_time=self.t_now,
            t_oldest=self.t_oldest,
            num_time_structures_in_a_week=self.num_time_structures_in_a_week,
        )
        self.K = int(np.ceil(K))

        # Define the model
        for i, h in enumerate(self.hidden_layer_size):
            if i == 0:
                input_size = self.dim_x + self.max_num_time_structures_after_t_oldest
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(
            ("softmax", nn.Softmax(dim=1))
        )  ######## here is the difference from regression based policy learner ########

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)

        # the length of the vector is the number of epochs
        self.train_loss = []
        self.train_value = []
        self.test_value = []
        # Create the vector that contains independent variables when we use linear regression X \in R^K
        self.X = np.arange(1, self.K + 1)
        # Map each element of X using given feature function to create Phi \in R^(K \times d)
        self.Phi = self.time_feature_func_for_Prognosticator_vec(
            self.X, self.num_parameters, self.max_num_time_structures_after_t_oldest
        )

        self.optimal_w = np.linalg.inv(self.Phi.T @ self.Phi) @ self.Phi.T
        self.optimal_w = torch.tensor(self.optimal_w, dtype=torch.float32)

    def convert_unix_time_to_num_time_structures_after_the_oldest_time_simple(
        self, unix_time
    ) -> int:
        return convert_unix_time_to_num_time_structures_after_the_oldest_time(
            unix_time=unix_time,
            t_oldest=self.t_oldest,
            num_time_structures_in_a_week=self.num_time_structures_in_a_week,
        )

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, t, a, r = (
            dataset["context"],
            dataset["time"],
            dataset["action"],
            dataset["reward"],
        )  ##### Added time for OPFV ######
        pscore, pi_0 = dataset["pscore"], np.squeeze(dataset["pi_b"])
        # Instatiate the solver (adagrad or adam)
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.evaluation_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.evaluation_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        # Create the training data loader
        training_data_loader = self._create_train_data_for_opl(
            x,
            t,
            a,
            r,
            pscore,
            pi_0,
        )

        # start policy training
        q_x_a_train, q_x_a_test = (
            dataset["expected_reward"],
            dataset_test["expected_reward"],
        )
        for _ in range(self.max_iter):
            if _ == 0:
                self.nn_model.eval()
                # Calcualte the trained policy using the training data
                pi_train = self.predict(dataset)
                # Calcualte the true value of the trained policy
                self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

                # Calcualte the trained policy using the test data
                pi_test = self.predict(dataset_test)
                # Calcualte the true value of the test policy
                self.test_value.append((q_x_a_test * pi_test).sum(1).mean())
            loss_epoch = 0.0
            # Start training mode
            self.nn_model.train()
            for x, t, a, r, p, pi_0_ in training_data_loader:
                optimizer.zero_grad()
                t_test_sampled = self.random_.choice(
                    self.time_test, size=self.num_time_learn, replace=True
                )
                convert_unix_time_to_num_time_structures_after_the_oldest_time_vec = (
                    np.vectorize(
                        convert_unix_time_to_num_time_structures_after_the_oldest_time
                    )
                )
                time_structures_t_test_sampled = convert_unix_time_to_num_time_structures_after_the_oldest_time_vec(
                    unix_time=t_test_sampled,
                    t_oldest=self.t_oldest,
                    num_time_structures_in_a_week=self.num_time_structures_in_a_week,
                )
                time_structures_t_test_sampled = torch.tensor(
                    time_structures_t_test_sampled
                )
                t_cloned = t.clone()
                t_cloned = t_cloned.to(torch.int)
                t_feature = t_cloned.apply_(
                    self.convert_unix_time_to_num_time_structures_after_the_oldest_time_simple
                )
                t_context = torch.tensor(
                    np.eye(self.max_num_time_structures_after_t_oldest)[t_feature - 1]
                )
                input = torch.cat((x, t_context), dim=1)
                input = input.to(torch.float32)
                pi = self.nn_model(input)

                loss = -self._estimate_policy_gradient(
                    x=x,
                    t=t,
                    t_test_sampled=t_test_sampled,
                    time_structures_t_test_sampled=time_structures_t_test_sampled,
                    a=a,
                    r=r,
                    pscore=p,
                    pi_0=pi_0_,
                    pi=pi,
                ).mean()
                loss.backward()
                # Update the parameters using the calculated gradient
                optimizer.step()
                loss_epoch += loss.item()

            # Calcualte the trained policy using the training data
            pi_train = self.predict(dataset)
            # Calcualte the true value of the trained policy
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

            # Calcualte the trained policy using the test data
            pi_test = self.predict(dataset_test)
            # Calcualte the true value of the test policy
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

            self.train_loss.append(loss_epoch)

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        t: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        pi_0: np.ndarray,
    ) -> tuple:
        dataset = Prognosticatordataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(t).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=torch.manual_seed(self.random_state),
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        x: torch.Tensor,
        t_test_sampled: torch.Tensor,
        time_structures_t_test_sampled: torch.Tensor,
        t: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
    ) -> torch.Tensor:
        current_pi_ai_xi_tj = pi.detach()
        idx = torch.arange(a.shape[0], dtype=torch.long)
        iw = current_pi_ai_xi_tj[idx, a] / pscore
        log_pi_ai_xi = torch.log(pi + self.log_eps)

        round_reward = iw * r * log_pi_ai_xi[idx, a]
        t_cloned = t.clone()
        t_feature = t_cloned.apply_(
            self.convert_unix_time_to_num_time_structures_after_the_oldest_time_simple
        )
        t_feature = t_feature.to(torch.int)
        mask = t_feature > self.K
        t_feature[mask] = self.K
        indicator_mat = torch.tensor(np.eye(self.K)[t_feature - 1], dtype=torch.float32)

        episode_wise_reward = indicator_mat.T @ round_reward

        res = self.optimal_w @ episode_wise_reward
        time_structures_t_test_sampled_cloned = time_structures_t_test_sampled.clone()
        time_structures_t_test_sampled_cloned_feature = (
            self.time_feature_func_for_Prognosticator_vec(
                time_structures_t_test_sampled_cloned,
                self.num_parameters,
                self.max_num_time_structures_after_t_oldest,
            )
        )
        time_structures_t_test_sampled_cloned_feature = torch.tensor(
            time_structures_t_test_sampled_cloned_feature, dtype=torch.float32
        )
        estimated_policy_grad_arr = time_structures_t_test_sampled_cloned_feature @ res

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float()
        t = torch.from_numpy(dataset_test["time"]).float()
        t_cloned = t.clone()
        t_cloned = t_cloned.to(torch.int)
        t_cloned_feature = t_cloned.apply_(
            self.convert_unix_time_to_num_time_structures_after_the_oldest_time_simple
        )
        t_cloned_feature_context = np.eye(self.max_num_time_structures_after_t_oldest)[
            t_cloned_feature - 1
        ]
        t_cloned_feature_context = torch.tensor(
            t_cloned_feature_context, dtype=torch.float32
        )
        input_for_nn_model = torch.cat((x, t_cloned_feature_context), dim=1)

        input_for_nn_model = input_for_nn_model.to(dtype=torch.float32)
        return self.nn_model(input_for_nn_model).detach().numpy()


@dataclass
class OPFVPolicyLearner:
    dim_x: int  # d_x
    num_actions: int  # |A|
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 32
    evaluation_rate_init: float = 0.01
    alpha: float = 1e-6  # weight decay used in optimizer
    imit_reg: float = 0.0
    log_eps: float = 1e-10  # to calculate the log the input of the log should not be zero so engineering trick
    solver: str = "adagrad"
    max_iter: int = 50  # number of epochs
    random_state: int = 12345
    phi_scalar_func_for_OPFV: callable = None  # Time structure for reward for OPFV
    num_time_structure: int = conf.num_true_time_structure_for_OPFV_reward
    time_test: np.ndarray = None
    reg_model_time: Any = None
    reg_model_time_struture: Any = (None,)
    num_time_learn: int = (conf.num_time_learn,)
    flag_tune: bool = (False,)
    phi_scalar_func_for_OPFV_list: list = None  # Time structure for reward for OPFV,

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.dim_x

        # Set the activation layer (Tanh, ReLU, or ELU)
        # Default ELU
        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        # Define the model
        for i, h in enumerate(self.hidden_layer_size):
            if i == 0:
                input_size = self.dim_x + self.num_time_structure
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(
            ("softmax", nn.Softmax(dim=1))
        )  ######## here is the difference from regression based policy learner ########

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)

        # the length of the vector is the number of epochs
        self.train_loss = []
        self.train_value = []
        self.test_value = []

    def fit(self, dataset: dict, dataset_test: dict, q_hat: np.ndarray = None) -> None:
        x, t, a, r = (
            dataset["context"],
            dataset["time"],
            dataset["action"],
            dataset["reward"],
        )  ##### Added time for OPFV ######
        pscore, pi_0 = dataset["pscore"], np.squeeze(dataset["pi_b"])
        # if \hat{q}(x, a) is not provided, then create zero vector
        if q_hat is None:
            q_hat = np.zeros((r.shape[0], self.num_actions))

        # Instatiate the solver (adagrad or adam)
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.evaluation_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.evaluation_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        # Create the training data loader
        training_data_loader = self._create_train_data_for_opl(
            x,
            t,
            a,
            r,
            pscore,
            q_hat,
            pi_0,
        )

        # start policy training
        q_x_a_train, q_x_a_test = (
            dataset["expected_reward"],
            dataset_test["expected_reward"],
        )
        for _ in tqdm(range(self.max_iter)):
            time_OPFV_fit_for_each_iter_start = time.time()
            if _ == 0:
                self.nn_model.eval()
                # Calcualte the trained policy using the training data
                pi_train = self.predict(dataset)
                # Calcualte the true value of the trained policy
                self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

                # Calcualte the trained policy using the test data
                pi_test = self.predict(dataset_test)
                # Calcualte the true value of the test policy
                self.test_value.append((q_x_a_test * pi_test).sum(1).mean())
            loss_epoch = 0.0
            # Start training mode
            self.nn_model.train()
            for x, t, a, r, p, q_hat_, pi_0_ in training_data_loader:
                optimizer.zero_grad()
                t_test_sampled = self.random_.choice(
                    self.time_test, size=self.num_time_learn, replace=True
                )
                phi_scalar_func_for_OPFV_vectorized = np.vectorize(
                    self.phi_scalar_func_for_OPFV
                )
                time_structure_t_test_sampled = phi_scalar_func_for_OPFV_vectorized(
                    t_test_sampled
                )
                t_test_sampled = torch.tensor(t_test_sampled)
                n = x.shape[0]
                m = t_test_sampled.shape[0]
                dim_x = x.shape[1]
                time_structure_t_test_sampled_augmented = np.tile(
                    time_structure_t_test_sampled, (n, 1)
                ).T
                one_hot_encoded_time_structure_t_test_sampled_augmented = np.eye(
                    self.num_time_structure
                )[time_structure_t_test_sampled_augmented]
                x_augmented = np.tile(x, (m, 1, 1))

                input_for_nn_model = np.concatenate(
                    (
                        x_augmented,
                        one_hot_encoded_time_structure_t_test_sampled_augmented,
                    ),
                    axis=2,
                )
                input_for_nn_model = torch.tensor(
                    input_for_nn_model, dtype=torch.float32
                )
                input_for_nn_model = input_for_nn_model.view(
                    n * m, dim_x + self.num_time_structure
                )
                pi = self.nn_model(input_for_nn_model)
                pi = pi.view(m, n, self.num_actions)
                if self.flag_tune == False:
                    loss = -self._estimate_policy_gradient(
                        x=x,
                        t=t,
                        t_test_sampled=t_test_sampled,
                        a=a,
                        r=r,
                        pscore=p,
                        f_hat=q_hat_,
                        pi_0=pi_0_,
                        pi=pi,
                    ).mean()
                    loss.backward()
                    optimizer.step()
                    loss_epoch += loss.item()
                else:
                    cardinality_Phi = len(self.phi_scalar_func_for_OPFV_list)
                    loss_candidate_vec = torch.zeros(cardinality_Phi)
                    approx_MSE_candidate_vec = torch.zeros(cardinality_Phi)
                    nabla_hat_V_vec_finest = -self._estimate_policy_gradient(
                        x=x,
                        t=t,
                        t_test_sampled=t_test_sampled,
                        a=a,
                        r=r,
                        pscore=p,
                        f_hat=q_hat_,
                        pi_0=pi_0_,
                        pi=pi,
                        phi_func=self.phi_scalar_func_for_OPFV_list[-1],
                    ).mean(axis=1)
                    for i in range(cardinality_Phi):
                        nabla_hat_V_mat = -self._estimate_policy_gradient(
                            x=x,
                            t=t,
                            t_test_sampled=t_test_sampled,
                            a=a,
                            r=r,
                            pscore=p,
                            f_hat=q_hat_,
                            pi_0=pi_0_,
                            pi=pi,
                            phi_func=self.phi_scalar_func_for_OPFV_list[i],
                        )
                        sample_variance_vec = nabla_hat_V_mat.var(axis=1)
                        nabla_hat_V_vec = nabla_hat_V_mat.mean(axis=1)
                        approx_bias_vec = abs(nabla_hat_V_vec - nabla_hat_V_vec_finest)
                        approx_MSE_vec = approx_bias_vec**2 + sample_variance_vec
                        approx_MSE_candidate_vec[i] = approx_MSE_vec.mean()
                        loss_candidate_vec[i] = nabla_hat_V_mat.mean()

                    index_for_tuned_phi = torch.argmin(approx_MSE_candidate_vec)
                    # print(f'index_for_tuned_phi = {index_for_tuned_phi}')
                    loss = loss_candidate_vec[index_for_tuned_phi]
                    loss.backward()
                    optimizer.step()
                    loss_epoch += loss.item()

            # Calcualte the trained policy using the training data
            pi_train = self.predict(dataset)
            # Calcualte the true value of the trained policy
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

            # Calcualte the trained policy using the test data
            pi_test = self.predict(dataset_test)
            # Calcualte the true value of the test policy
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

            self.train_loss.append(loss_epoch)

            time_OPFV_fit_for_each_iter_end = time.time()

            execution_time_OPFV_fit_for_each_iter = (
                time_OPFV_fit_for_each_iter_end - time_OPFV_fit_for_each_iter_start
            )
            if conf.flag_show_execution_time_for_each_iter_in_OPFV == True:
                print(
                    f"Execution time for iter {_ + 1}/{self.max_iter} in OPFV = {execution_time_OPFV_fit_for_each_iter / 60:.3f} mins"
                )

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        t: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        pi_0: np.ndarray,
    ) -> tuple:
        dataset = OPFVDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(t).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        x: torch.Tensor,
        t_test_sampled: torch.Tensor,
        t: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        f_hat: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
        phi_func: callable = None,
    ) -> torch.Tensor:
        n = x.shape[0]
        m = t_test_sampled.shape[0]

        # print(f'n = {n}')
        # print(f'm = {m}')

        current_pi_ai_xi_tj = pi.detach()
        idx = torch.arange(a.shape[0], dtype=torch.long)
        current_pi_pscore = current_pi_ai_xi_tj[:, idx, a]
        iw_matrix = current_pi_pscore / pscore

        # print(f'shape of current_pi_ai_xi_tj = {current_pi_ai_xi_tj.shape}')
        # print(f'shape of current_pi_pscore = {current_pi_pscore.shape}')
        # print(f'shape of iw_matrix = {iw_matrix.shape}')

        log_pi_ai_xi_tj = torch.log(pi + self.log_eps)
        log_pi_ai_xi_tj_factual = log_pi_ai_xi_tj[:, idx, a]
        # print(f'shape of log_pi_ai_xi_tj = {log_pi_ai_xi_tj.shape}')
        # print(f'shape of log_pi_ai_xi_tj_factual = {log_pi_ai_xi_tj_factual.shape}')

        if phi_func == None:
            phi_vector_func_for_reward = np.vectorize(self.phi_scalar_func_for_OPFV)
        else:
            phi_vector_func_for_reward = np.vectorize(phi_func)
        time_structure_for_reward_test = torch.tensor(
            phi_vector_func_for_reward(t_test_sampled)
        )
        time_structure_for_reward_train = torch.tensor(phi_vector_func_for_reward(t))
        time_structure_for_reward_test_unsqueezed = (
            time_structure_for_reward_test.unsqueeze(0)
        )
        time_structure_for_reward_train_unsqueezed = (
            time_structure_for_reward_train.unsqueeze(1)
        )
        # print(f'shape of time_structure_for_reward_test = {time_structure_for_reward_test.shape}')
        # print(f'shape of time_structure_for_reward_train = {time_structure_for_reward_train.shape}')
        # print(f'shape of time_structure_for_reward_test_unsqueezed = {time_structure_for_reward_test_unsqueezed.shape}')
        # print(f'shape of time_structure_for_reward_train_unsqueezed = {time_structure_for_reward_train_unsqueezed.shape}')

        indicator_phi_r_matrix = (
            (
                time_structure_for_reward_test_unsqueezed
                == time_structure_for_reward_train_unsqueezed
            )
            .float()
            .T
        )
        # print(f'shape of indicator_phi_r_matrix = {indicator_phi_r_matrix.shape}')

        P_phi_tj_vec = torch.mean(indicator_phi_r_matrix, dim=1)
        P_phi_tj_vec[P_phi_tj_vec == 0] = 0.000001
        # print(f'shape of P_phi_tj_vec = {P_phi_tj_vec.shape}')

        ind_div_P_phi_matrix = indicator_phi_r_matrix / P_phi_tj_vec[:, None]
        # print(f'shape of ind_div_P_phi_matrix = {ind_div_P_phi_matrix.shape}')

        f_hat = f_hat.squeeze(2)
        # print(f'shape of f_hat = {f_hat.shape}')

        f_hat_xi_ti_ai_factual_vec = f_hat[idx, a]
        # print(f'shape of f_hat_xi_ti_ai_factual_vec = {f_hat_xi_ti_ai_factual_vec.shape}')

        estimated_policy_grad_arr = (
            ind_div_P_phi_matrix
            * iw_matrix
            * (r - f_hat_xi_ti_ai_factual_vec)
            * log_pi_ai_xi_tj_factual
        )
        # print(f'shape of estimated_policy_grad_arr = {estimated_policy_grad_arr.shape}')

        if self.reg_model_time != None:
            pass
        else:
            time_structure_input_for_reg_model_time = (
                time_structure_for_reward_test.repeat_interleave(n)
            )
            x_input_for_reg_model_time = x.repeat(m, 1)
            f_xi_tj_ai = self.reg_model_time_struture.predict(
                context=x_input_for_reg_model_time,
                time_structure=time_structure_input_for_reg_model_time,
            )
        f_xi_tj_ai = np.squeeze(f_xi_tj_ai, axis=2)
        f_xi_tj_ai = torch.tensor(f_xi_tj_ai, requires_grad=True)
        f_xi_tj_ai = f_xi_tj_ai.view(m, n, self.num_actions)
        # print(f'shape of f_xi_tj_ai = {f_xi_tj_ai.shape}')

        estimated_policy_grad_arr += torch.sum(
            current_pi_ai_xi_tj * f_xi_tj_ai * log_pi_ai_xi_tj, dim=2
        )

        # print(f'shape of estimated_policy_grad_arr = {estimated_policy_grad_arr.shape}')
        # print(f'-estimated_policy_grad_arr.mean() = {-estimated_policy_grad_arr.mean()}')
        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float()
        t = torch.from_numpy(dataset_test["time"]).float()
        t_cloned2 = t.clone()
        one_of_the_inputs2 = t_cloned2.apply_(self.phi_scalar_func_for_OPFV).to(
            dtype=torch.int32
        )
        one_of_the_inputs2 = np.eye(self.num_time_structure)[one_of_the_inputs2]
        one_of_the_inputs2 = torch.tensor(one_of_the_inputs2)
        input_for_nn_model = torch.cat((x, one_of_the_inputs2), dim=1)
        input_for_nn_model = input_for_nn_model.to(dtype=torch.float32)
        return self.nn_model(input_for_nn_model).detach().numpy()
