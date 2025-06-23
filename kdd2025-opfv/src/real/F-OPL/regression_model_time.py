# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state, check_scalar


@dataclass
class RegressionModelTime(BaseEstimator):
    base_model: BaseEstimator
    n_actions: int
    len_list: int = 1
    action_context: Optional[np.ndarray] = None
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if not (
            isinstance(self.fitting_method, str)
            and self.fitting_method in ["normal", "iw", "mrdr"]
        ):
            raise ValueError(
                f"`fitting_method` must be one of 'normal', 'iw', or 'mrdr', but {self.fitting_method} is given"
            )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "`base_model` must be BaseEstimator or a child class of BaseEstimator"
            )

        self.base_model_list = [
            clone(self.base_model) for _ in np.arange(self.len_list)
        ]
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)

    def fit(
        self,
        context: np.ndarray,
        time: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
    ) -> None:
        n = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
            if not np.allclose(action_dist.sum(axis=1), 1):
                raise ValueError("`action_dist` must be a probability distribution")
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            X = self._pre_process_for_reg_model(
                context=context[idx],
                time=time[idx],
                action=action[idx],
                action_context=self.action_context,
            )
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {pos_}")
            # train the base model according to the given `fitting method`
            if self.fitting_method == "normal":
                self.base_model_list[pos_].fit(X, reward[idx])
            else:
                action_dist_at_pos = action_dist[np.arange(n), action, pos_][idx]
                if self.fitting_method == "iw":
                    sample_weight = action_dist_at_pos / pscore[idx]
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )
                elif self.fitting_method == "mrdr":
                    sample_weight = action_dist_at_pos
                    sample_weight *= 1.0 - pscore[idx]
                    sample_weight /= pscore[idx] ** 2
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )

    def predict(self, context: np.ndarray, time: np.ndarray) -> np.ndarray:
        n = context.shape[0]
        q_hat = np.zeros((n, self.n_actions, self.len_list))
        for action_ in np.arange(self.n_actions):
            for pos_ in np.arange(self.len_list):
                X = self._pre_process_for_reg_model(
                    context=context,
                    time=time,
                    action=action_ * np.ones(n, int),
                    action_context=self.action_context,
                )
                q_hat_ = (
                    self.base_model_list[pos_].predict_proba(X)[:, 1]
                    if is_classifier(self.base_model_list[pos_])
                    else self.base_model_list[pos_].predict(X)
                )
                q_hat[np.arange(n), action_, pos_] = q_hat_
        return q_hat

    def fit_predict(
        self,
        context: np.ndarray,
        time: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
        n_folds: int = 3,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        n_rounds = context.shape[0]

        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        if n_folds == 1:
            self.fit(
                context=context,
                time=time,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                action_dist=action_dist,
            )
            return self.predict(context=context, time=time)
        else:
            q_hat = np.zeros((n_rounds, self.n_actions, self.len_list))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        for train_idx, test_idx in kf.split(context):
            action_dist_tr = (
                action_dist[train_idx] if action_dist is not None else action_dist
            )
            self.fit(
                context=context[train_idx],
                time=time[train_idx],
                action=action[train_idx],
                reward=reward[train_idx],
                pscore=pscore[train_idx],
                position=position[train_idx],
                action_dist=action_dist_tr,
            )
            q_hat[test_idx, :, :] = self.predict(
                context=context[test_idx], time=time[test_idx]
            )
        return q_hat

    def _pre_process_for_reg_model(
        self,
        context: np.ndarray,
        time: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
    ) -> np.ndarray:
        return np.c_[context, time, action_context[action]]


@dataclass
class RegressionModelTimeStructure(BaseEstimator):
    base_model: BaseEstimator
    n_actions: int
    len_list: int = 1
    action_context: Optional[np.ndarray] = None
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if not (
            isinstance(self.fitting_method, str)
            and self.fitting_method in ["normal", "iw", "mrdr"]
        ):
            raise ValueError(
                f"`fitting_method` must be one of 'normal', 'iw', or 'mrdr', but {self.fitting_method} is given"
            )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "`base_model` must be BaseEstimator or a child class of BaseEstimator"
            )

        self.base_model_list = [
            clone(self.base_model) for _ in np.arange(self.len_list)
        ]
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)

    def fit(
        self,
        context: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
    ) -> None:
        n = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
            if not np.allclose(action_dist.sum(axis=1), 1):
                raise ValueError("`action_dist` must be a probability distribution")
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            X = self._pre_process_for_reg_model(
                context=context[idx],
                time_structure=time_structure[idx],
                action=action[idx],
                action_context=self.action_context,
            )
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {pos_}")
            # train the base model according to the given `fitting method`
            if self.fitting_method == "normal":
                self.base_model_list[pos_].fit(X, reward[idx])
            else:
                action_dist_at_pos = action_dist[np.arange(n), action, pos_][idx]
                if self.fitting_method == "iw":
                    sample_weight = action_dist_at_pos / pscore[idx]
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )
                elif self.fitting_method == "mrdr":
                    sample_weight = action_dist_at_pos
                    sample_weight *= 1.0 - pscore[idx]
                    sample_weight /= pscore[idx] ** 2
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )

    def predict(self, context: np.ndarray, time_structure: np.ndarray) -> np.ndarray:
        n = context.shape[0]
        q_hat = np.zeros((n, self.n_actions, self.len_list))
        for action_ in np.arange(self.n_actions):
            for pos_ in np.arange(self.len_list):
                X = self._pre_process_for_reg_model(
                    context=context,
                    time_structure=time_structure,
                    action=action_ * np.ones(n, int),
                    action_context=self.action_context,
                )
                q_hat_ = (
                    self.base_model_list[pos_].predict_proba(X)[:, 1]
                    if is_classifier(self.base_model_list[pos_])
                    else self.base_model_list[pos_].predict(X)
                )
                q_hat[np.arange(n), action_, pos_] = q_hat_
        return q_hat

    def fit_predict(
        self,
        context: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
        n_folds: int = 3,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        n_rounds = context.shape[0]

        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        if n_folds == 1:
            self.fit(
                context=context,
                time_structure=time_structure,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                action_dist=action_dist,
            )
            return self.predict(context=context, time_structure=time_structure)
        else:
            q_hat = np.zeros((n_rounds, self.n_actions, self.len_list))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        for train_idx, test_idx in kf.split(context):
            action_dist_tr = (
                action_dist[train_idx] if action_dist is not None else action_dist
            )
            self.fit(
                context=context[train_idx],
                time_structure=time_structure[train_idx],
                action=action[train_idx],
                reward=reward[train_idx],
                pscore=pscore[train_idx],
                position=position[train_idx],
                action_dist=action_dist_tr,
            )
            q_hat[test_idx, :, :] = self.predict(
                context=context[test_idx], time_structure=time_structure[test_idx]
            )
        return q_hat

    def _pre_process_for_reg_model(
        self,
        context: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
    ) -> np.ndarray:
        return np.c_[context, time_structure, action_context[action]]


@dataclass
class RegressionModelTimeTrue(BaseEstimator):
    base_model: BaseEstimator
    n_actions: int
    len_list: int = 1
    action_context: Optional[np.ndarray] = None
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if not (
            isinstance(self.fitting_method, str)
            and self.fitting_method in ["normal", "iw", "mrdr"]
        ):
            raise ValueError(
                f"`fitting_method` must be one of 'normal', 'iw', or 'mrdr', but {self.fitting_method} is given"
            )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "`base_model` must be BaseEstimator or a child class of BaseEstimator"
            )

        self.base_model_list = [
            clone(self.base_model) for _ in np.arange(self.len_list)
        ]
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)

    def fit(
        self,
        context: np.ndarray,
        time: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
    ) -> None:
        n = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
            if not np.allclose(action_dist.sum(axis=1), 1):
                raise ValueError("`action_dist` must be a probability distribution")
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            X = self._pre_process_for_reg_model(
                context=context[idx],
                time=time[idx],
                time_structure=time_structure[idx],
                action=action[idx],
                action_context=self.action_context,
            )
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {pos_}")
            # train the base model according to the given `fitting method`
            if self.fitting_method == "normal":
                self.base_model_list[pos_].fit(X, reward[idx])
            else:
                action_dist_at_pos = action_dist[np.arange(n), action, pos_][idx]
                if self.fitting_method == "iw":
                    sample_weight = action_dist_at_pos / pscore[idx]
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )
                elif self.fitting_method == "mrdr":
                    sample_weight = action_dist_at_pos
                    sample_weight *= 1.0 - pscore[idx]
                    sample_weight /= pscore[idx] ** 2
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )

    def predict(
        self, context: np.ndarray, time: np.ndarray, time_structure: np.ndarray
    ) -> np.ndarray:
        n = context.shape[0]
        q_hat = np.zeros((n, self.n_actions, self.len_list))
        for action_ in np.arange(self.n_actions):
            for pos_ in np.arange(self.len_list):
                X = self._pre_process_for_reg_model(
                    context=context,
                    time=time,
                    time_structure=time_structure,
                    action=action_ * np.ones(n, int),
                    action_context=self.action_context,
                )
                q_hat_ = (
                    self.base_model_list[pos_].predict_proba(X)[:, 1]
                    if is_classifier(self.base_model_list[pos_])
                    else self.base_model_list[pos_].predict(X)
                )
                q_hat[np.arange(n), action_, pos_] = q_hat_
        return q_hat

    def fit_predict(
        self,
        context: np.ndarray,
        time: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
        n_folds: int = 3,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        n_rounds = context.shape[0]

        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        if n_folds == 1:
            self.fit(
                context=context,
                time=time,
                time_structure=time_structure,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                action_dist=action_dist,
            )
            return self.predict(
                context=context, time=time, time_structure=time_structure
            )
        else:
            q_hat = np.zeros((n_rounds, self.n_actions, self.len_list))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        for train_idx, test_idx in kf.split(context):
            action_dist_tr = (
                action_dist[train_idx] if action_dist is not None else action_dist
            )
            self.fit(
                context=context[train_idx],
                time=time[train_idx],
                time_structure=time_structure[train_idx],
                action=action[train_idx],
                reward=reward[train_idx],
                pscore=pscore[train_idx],
                position=position[train_idx],
                action_dist=action_dist_tr,
            )
            q_hat[test_idx, :, :] = self.predict(
                context=context[test_idx],
                time=time[test_idx],
                time_structure=time_structure[test_idx],
            )
        return q_hat

    def _pre_process_for_reg_model(
        self,
        context: np.ndarray,
        time: np.ndarray,
        time_structure: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
    ) -> np.ndarray:
        return np.c_[context, time, time_structure, action_context[action]]
