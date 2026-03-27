# Copyright (c) 2026 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from obp.ope import SlateOffPolicyEvaluation


@dataclass
class OffPolicyEvaluation(SlateOffPolicyEvaluation):
    def estimate_policy_values(
        self,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        evaluation_policy_p_click: Optional[np.ndarray] = None,
        behavior_policy_p_click: Optional[np.ndarray] = None,
        estimated_conversion_factual: Optional[np.ndarray] = None,
        estimated_behavior_policy_p_click: Optional[np.ndarray] = None,
        estimated_evaluation_policy_p_click: Optional[np.ndarray] = None,
        q_hat_by_estimated_click_model: Optional[np.ndarray] = None,
        dm_term: Optional[np.ndarray] = None,
        dm_term_by_click_model: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:

        policy_value_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            evaluation_policy_pscore=evaluation_policy_pscore,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            q_hat=q_hat,
            evaluation_policy_p_click=evaluation_policy_p_click,
            behavior_policy_p_click=behavior_policy_p_click,
            estimated_conversion_factual=estimated_conversion_factual,
            estimated_behavior_policy_p_click=estimated_behavior_policy_p_click,
            estimated_evaluation_policy_p_click=estimated_evaluation_policy_p_click,
            q_hat_by_estimated_click_model=q_hat_by_estimated_click_model,
            dm_term=dm_term,
            dm_term_by_click_model=dm_term_by_click_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_dict[estimator_name] = estimator.estimate_policy_value(
                **estimator_inputs
            )

        return policy_value_dict

    def _create_estimator_inputs(
        self,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        evaluation_policy_p_click: Optional[np.ndarray] = None,
        behavior_policy_p_click: Optional[np.ndarray] = None,
        estimated_conversion_factual: Optional[np.ndarray] = None,
        estimated_behavior_policy_p_click: Optional[np.ndarray] = None,
        estimated_evaluation_policy_p_click: Optional[np.ndarray] = None,
        q_hat_by_estimated_click_model: Optional[np.ndarray] = None,
        dm_term: Optional[np.ndarray] = None,
        dm_term_by_click_model: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Create input dictionary to estimate policy value by subclasses of `BaseSlateOffPolicyEstimator`"""
        if (
            evaluation_policy_pscore is None
            and evaluation_policy_pscore_item_position is None
            and evaluation_policy_pscore_cascade is None
        ):
            raise ValueError(
                "one of `evaluation_policy_pscore`, `evaluation_policy_pscore_item_position`, or `evaluation_policy_pscore_cascade` must be given"
            )

        estimator_inputs = {
            input_: self.bandit_feedback[input_]
            for input_ in [
                "slate_id",
                "action",
                "reward",
                "position",
                "pscore",
                "pscore_item_position",
                "pscore_cascade",
            ]
            if input_ in self.bandit_feedback
        }
        estimator_inputs["evaluation_policy_pscore"] = evaluation_policy_pscore
        estimator_inputs["evaluation_policy_pscore_item_position"] = (
            evaluation_policy_pscore_item_position
        )
        estimator_inputs["evaluation_policy_pscore_cascade"] = (
            evaluation_policy_pscore_cascade
        )
        estimator_inputs["evaluation_policy_action_dist"] = (
            evaluation_policy_action_dist
        )
        estimator_inputs["q_hat"] = q_hat

        estimator_inputs["evaluation_policy_p_click"] = evaluation_policy_p_click
        estimator_inputs["behavior_policy_p_click"] = behavior_policy_p_click

        estimator_inputs["estimated_conversion_factual"] = estimated_conversion_factual

        estimator_inputs["estimated_behavior_policy_p_click"] = (
            estimated_behavior_policy_p_click
        )
        estimator_inputs["estimated_evaluation_policy_p_click"] = (
            estimated_evaluation_policy_p_click
        )
        estimator_inputs["q_hat_by_estimated_click_model"] = (
            q_hat_by_estimated_click_model
        )

        estimator_inputs["dm_term"] = dm_term
        estimator_inputs["dm_term_by_click_model"] = dm_term_by_click_model
        return estimator_inputs
