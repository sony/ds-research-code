# Copyright (c) 2026 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
"""Off-Policy Estimators for Slate/Ranking Policies."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from obp.ope.estimators_slate import (
    BaseSlateOffPolicyEstimator,
)
from obp.utils import check_sips_inputs, estimate_confidence_interval_by_bootstrap
from sklearn.utils import check_scalar


@dataclass
class BaseSlateInverseProbabilityWeighting(BaseSlateOffPolicyEstimator):
    len_list: int

    def _estimate_slate_confidence_interval_by_bootstrap(
        self,
        slate_id: np.ndarray,
        estimated_rewards: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:

        unique_slate = np.unique(slate_id)
        # sum estimated_rewards in each slate
        estimated_round_rewards = list()
        for slate in unique_slate:
            estimated_round_rewards.append(estimated_rewards[slate_id == slate].sum())
        estimated_round_rewards = np.array(estimated_round_rewards)
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ClickBasedIPS(BaseSlateInverseProbabilityWeighting):
    estimator_name: str = "cips"
    use_estimated_click_model: bool = False

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> np.ndarray:

        iw = evaluation_policy_pscore / behavior_policy_pscore
        estimated_rewards = iw * reward

        return estimated_rewards

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_p_click: np.ndarray,
        evaluation_policy_p_click: np.ndarray,
        estimated_behavior_policy_p_click: Optional[np.ndarray] = None,
        estimated_evaluation_policy_p_click: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:

        if self.use_estimated_click_model == True:
            return (
                self._estimate_round_rewards(
                    reward=reward,
                    position=position,
                    behavior_policy_pscore=estimated_behavior_policy_p_click,
                    evaluation_policy_pscore=estimated_evaluation_policy_p_click,
                ).sum()
                / np.unique(slate_id).shape[0]
            )
        else:
            return (
                self._estimate_round_rewards(
                    reward=reward,
                    position=position,
                    behavior_policy_pscore=behavior_policy_p_click,
                    evaluation_policy_pscore=evaluation_policy_p_click,
                ).sum()
                / np.unique(slate_id).shape[0]
            )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:

        check_sips_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ClickBasedDR(BaseSlateInverseProbabilityWeighting):
    estimator_name: str = "cdr"
    use_estimated_click_model: bool = False

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        q_hat: Optional[np.ndarray] = None,
        estimated_conversion_factual: Optional[np.ndarray] = None,
        dm_term: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:

        iw = evaluation_policy_pscore / behavior_policy_pscore
        estimated_rewards = iw * (reward - q_hat)
        # estimated_rewards += evaluation_policy_pscore*estimated_conversion_factual
        estimated_rewards = np.append(estimated_rewards, dm_term)

        return estimated_rewards

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_p_click: np.ndarray,
        evaluation_policy_p_click: np.ndarray,
        q_hat: np.ndarray,
        estimated_conversion_factual: np.ndarray,
        estimated_behavior_policy_p_click: Optional[np.ndarray] = None,
        estimated_evaluation_policy_p_click: Optional[np.ndarray] = None,
        q_hat_by_estimated_click_model: Optional[np.ndarray] = None,
        dm_term: Optional[np.ndarray] = None,
        dm_term_by_click_model: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:

        if self.use_estimated_click_model == True:
            return (
                self._estimate_round_rewards(
                    reward=reward,
                    position=position,
                    behavior_policy_pscore=estimated_behavior_policy_p_click,
                    evaluation_policy_pscore=estimated_evaluation_policy_p_click,
                    q_hat=q_hat_by_estimated_click_model,
                    estimated_conversion_factual=estimated_conversion_factual,
                    dm_term=dm_term_by_click_model,
                ).sum()
                / np.unique(slate_id).shape[0]
            )
        else:
            return (
                self._estimate_round_rewards(
                    reward=reward,
                    position=position,
                    behavior_policy_pscore=behavior_policy_p_click,
                    evaluation_policy_pscore=evaluation_policy_p_click,
                    q_hat=q_hat,
                    estimated_conversion_factual=estimated_conversion_factual,
                    dm_term=dm_term,
                ).sum()
                / np.unique(slate_id).shape[0]
            )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:

        check_sips_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
