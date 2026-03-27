# Copyright (c) 2026 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import obp
import pandas as pd
import seaborn as sns
from dataset import SyntheticSlateBanditDataset, linear_behavior_policy_logit
from estimator import (
    ClickBasedDR as CDR,
)
from estimator import (
    ClickBasedIPS as CIPS,
)
from obp.dataset import (
    linear_behavior_policy,
    linear_reward_function,
    logistic_reward_function,
)
from obp.ope import (
    # SlateOffPolicyEvaluation,
    RegressionModel,
)
from obp.ope import (
    SlateIndependentIPS as IIPS,
)
from obp.ope import (
    SlateRewardInteractionIPS as RIPS,
)
from obp.ope import (
    SlateStandardIPS as IPS,
)
from omegaconf import DictConfig, OmegaConf
from ope import OffPolicyEvaluation
from pandas import DataFrame
from plot import (
    plot,
    plot_normalize,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from tqdm import tqdm


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    if cfg.setting.deterministic_user_threshold == "-inf":
        cfg.setting.deterministic_user_threshold = -np.inf
    if cfg.setting.deterministic_user_threshold == "inf":
        cfg.setting.deterministic_user_threshold = np.inf

    np.random.seed(cfg.setting.random_state)
    num_runs = cfg.setting.num_runs
    num_data_list = cfg.setting.num_data_list

    if cfg.setting.reward_type_conversion == "continuous":
        dataset = SyntheticSlateBanditDataset(
            n_unique_action=cfg.setting.n_unique_action,
            len_list=cfg.setting.len_list,
            dim_context=cfg.setting.dim_context,
            reward_type=cfg.setting.reward_type,
            reward_structure=cfg.setting.reward_structure,
            decay_function=cfg.setting.decay_function,
            base_reward_function=logistic_reward_function,
            base_reward_function_conversion=linear_reward_function,
            behavior_policy_function=linear_behavior_policy_logit,
            is_factorizable=cfg.setting.is_factorizable,
            random_state=cfg.setting.random_state,
            reward_type_conversion=cfg.setting.reward_type_conversion,
            reward_structure_conversion=cfg.setting.reward_structure_conversion,
            deterministic_user_threshold=cfg.setting.deterministic_user_threshold,
            effect_from_ranking=cfg.setting.effect_from_ranking,
        )
    else:  # binary
        dataset = SyntheticSlateBanditDataset(
            n_unique_action=cfg.setting.n_unique_action,
            len_list=cfg.setting.len_list,
            dim_context=cfg.setting.dim_context,
            reward_type=cfg.setting.reward_type,
            reward_structure=cfg.setting.reward_structure,
            decay_function=cfg.setting.decay_function,
            base_reward_function=logistic_reward_function,
            base_reward_function_conversion=logistic_reward_function,
            behavior_policy_function=linear_behavior_policy_logit,
            is_factorizable=cfg.setting.is_factorizable,
            random_state=cfg.setting.random_state,
            reward_type_conversion=cfg.setting.reward_type_conversion,
            reward_structure_conversion=cfg.setting.reward_structure_conversion,
            deterministic_user_threshold=cfg.setting.deterministic_user_threshold,
            effect_from_ranking=cfg.setting.effect_from_ranking,
        )

    # evaluation policy
    n_test = cfg.setting.n_test
    context = np.random.normal(size=(n_test, cfg.setting.dim_context))

    if cfg.setting.evaluation_policy_logit == "linear_reward_function":
        evaluation_policy_logit = linear_reward_function(
            context=context,
            action_context=np.eye(cfg.setting.n_unique_action, dtype=int),
            random_state=cfg.setting.random_state,
        )
    else:
        evaluation_policy_logit = linear_behavior_policy_logit(
            context=context,
            action_context=np.eye(cfg.setting.n_unique_action, dtype=int),
            random_state=cfg.setting.random_state,
            tau=cfg.setting.tau_pi_e,
        )

    pi_e_value = dataset.calc_ground_truth_policy_value_epsilon_greedy(
        context=context,
        evaluation_policy_logit_=evaluation_policy_logit,
        eps=cfg.setting.eps,
    )
    print("pi_e_value", pi_e_value)

    bandit_data = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_test,
        # clip_logit_value=700.0,
    )
    print("pi_0_value", bandit_data["reward"].sum() / n_test)

    true_pi_0_value = bandit_data["reward"].sum() / n_test

    result_df_list = []
    for num_data in num_data_list:
        estimated_policy_value_list = []
        estimated_pi_0_value_list = []
        for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
            validation_bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=num_data,
            )

            if cfg.setting.evaluation_policy_logit == "linear_reward_function":
                evaluation_policy_logit = linear_reward_function(
                    context=validation_bandit_data["context"],
                    action_context=validation_bandit_data["action_context"],
                    random_state=cfg.setting.random_state,
                )
            else:
                evaluation_policy_logit = linear_behavior_policy_logit(
                    context=validation_bandit_data["context"],
                    action_context=validation_bandit_data["action_context"],
                    random_state=cfg.setting.random_state,
                    tau=cfg.setting.tau_pi_e,
                )

            (
                evaluation_policy_pscore,
                evaluation_policy_pscore_item_position,
                evaluation_policy_pscore_cascade,
                evaluation_policy_p_click,
                p_click_pi_e,
            ) = dataset.obtain_pscore_given_evaluation_policy_logit_epsilon_greedy(
                context=validation_bandit_data["context"],
                action=validation_bandit_data["action"],
                evaluation_policy_logit_=evaluation_policy_logit,
                eps=cfg.setting.eps,
            )

            # obtain regression model
            click_probability_true = validation_bandit_data[
                "expected_reward_factual_click"
            ]  # p_c(x,a_A)
            ################################################
            reg_model = RegressionModel(
                n_actions=cfg.setting.n_unique_action,
                base_model=MLPRegressor(
                    hidden_layer_sizes=(30, 30, 30),
                    max_iter=3000,
                    early_stopping=True,
                    random_state=cfg.setting.random_state,
                ),
            )
            mask = validation_bandit_data["reward_click"] == 1
            reg_model.fit(
                context=np.repeat(
                    validation_bandit_data["context"], dataset.len_list, axis=0
                )[mask],  # context; x
                action=validation_bandit_data["action"][mask],  # action; a
                reward=validation_bandit_data["reward"][mask],  # reward; r
            )
            # estimated_conversion (n_rounds*len_list, n_unique_actions, 1)
            estimated_conversion = reg_model.predict(
                context=np.repeat(
                    validation_bandit_data["context"], dataset.len_list, axis=0
                )
            )

            estimated_conversion_for_dm_term = reg_model.predict(
                context=validation_bandit_data["context"]
            )[:, :, 0]

            estimated_conversion_factual = estimated_conversion[
                np.arange(
                    dataset.len_list * validation_bandit_data["context"].shape[0]
                ),
                validation_bandit_data["action"],
                0,
            ]

            estimated_CR_factual = (
                click_probability_true * estimated_conversion_factual
            )  # true_click * estimated conversion
            ################################################
            ################################################
            # estimate click probability
            click_model = MLPClassifier(
                hidden_layer_sizes=(30, 30, 30),
                max_iter=3000,
                early_stopping=True,
                random_state=cfg.setting.random_state,
            )
            X_train = np.concatenate(
                [
                    validation_bandit_data["context"],
                    validation_bandit_data["action"].reshape(-1, dataset.len_list),
                ],
                axis=1,
            )
            y_train = validation_bandit_data["reward_click"].reshape(
                -1, dataset.len_list
            )
            click_model.fit(
                X=X_train,
                y=y_train,
            )

            (
                estimated_behavior_policy_p_click,
                estimated_evaluation_policy_p_click,
                p_click_pi_e_by_click_model,  # p_c(x,a,pi_e)
            ) = dataset.obtain_p_click_pi_given_estimated_click_probability(
                context=validation_bandit_data["context"],
                action=validation_bandit_data["action"],
                click_model=click_model,
                evaluation_policy_logit_type=cfg.setting.evaluation_policy_logit,
                eps=cfg.setting.eps,
                tau=cfg.setting.tau_pi_e,
            )
            click_probability_factual_by_click_model = click_model.predict_proba(
                X_train
            ).reshape(validation_bandit_data["action"].shape[0])
            estimated_CR_factual_by_click_model = (
                click_probability_factual_by_click_model * estimated_conversion_factual
            )  # true_click * estimated conversion

            dm_term = (p_click_pi_e * estimated_conversion_for_dm_term).sum()
            dm_term_by_click_model = (
                p_click_pi_e_by_click_model * estimated_conversion_for_dm_term
            ).sum()

            ################################################

            ope = OffPolicyEvaluation(
                bandit_feedback=validation_bandit_data,
                ope_estimators=[
                    IPS(estimator_name="IPS", len_list=cfg.setting.len_list),
                    IIPS(estimator_name="IIPS", len_list=cfg.setting.len_list),
                    RIPS(estimator_name="RIPS", len_list=cfg.setting.len_list),
                    CIPS(estimator_name="CIPS", len_list=cfg.setting.len_list),
                    CDR(estimator_name="CDR", len_list=cfg.setting.len_list),
                    CIPS(
                        estimator_name="CIPS (estimate)",
                        len_list=cfg.setting.len_list,
                        use_estimated_click_model=True,
                    ),
                    CDR(
                        estimator_name="CDR (estimate)",
                        len_list=cfg.setting.len_list,
                        use_estimated_click_model=True,
                    ),
                ],
            )

            estimated_policy_values = ope.estimate_policy_values(
                evaluation_policy_pscore=evaluation_policy_pscore,
                evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
                evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
                evaluation_policy_p_click=evaluation_policy_p_click,  # pc(x,a,pi)
                behavior_policy_p_click=validation_bandit_data[
                    "p_click_factual_pi_0"
                ],  # pc(x,a,pi_0)
                estimated_conversion_factual=estimated_conversion_factual,  # p_r
                q_hat=estimated_CR_factual,  # q_hat
                estimated_behavior_policy_p_click=estimated_behavior_policy_p_click,
                estimated_evaluation_policy_p_click=estimated_evaluation_policy_p_click,
                q_hat_by_estimated_click_model=estimated_CR_factual_by_click_model,
                dm_term=dm_term,
                dm_term_by_click_model=dm_term_by_click_model,
            )
            estimated_policy_value_list.append(estimated_policy_values)

            for _ in range(7):
                estimated_pi_0_value_list.append(
                    validation_bandit_data["reward"].sum() / num_data
                )

        # summarize result
        result_df = (
            DataFrame(DataFrame(estimated_policy_value_list).stack())
            .reset_index(1)
            .rename(columns={"level_1": "est", 0: "value"})
        )
        result_df["num_data"] = num_data
        result_df["pi_e_value"] = pi_e_value
        result_df["pi_0_value"] = true_pi_0_value
        result_df["estimated_pi_0_value"] = estimated_pi_0_value_list
        result_df["true_selection"] = (
            (result_df["pi_e_value"] - result_df["pi_0_value"])
            * (result_df["value"] - result_df["estimated_pi_0_value"])
            >= 0
        ).astype(int)
        result_df["se"] = (result_df.value - pi_e_value) ** 2
        result_df["bias"] = 0.0
        result_df["variance"] = 0.0

        sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
        for est_ in sample_mean["est"]:
            estimates = result_df.loc[result_df["est"] == est_, "value"].values
            mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
            mean_estimates = np.ones_like(estimates) * mean_estimates
            result_df.loc[result_df["est"] == est_, "bias"] = (
                pi_e_value - mean_estimates
            ) ** 2
            result_df.loc[result_df["est"] == est_, "variance"] = (
                estimates - mean_estimates
            ) ** 2
        result_df_list.append(result_df)
        print(
            "max_iw",
            (evaluation_policy_pscore / validation_bandit_data["pscore"]).max(),
        )
        print(
            "max_iw_CIPS",
            (
                evaluation_policy_p_click
                / validation_bandit_data["p_click_factual_pi_0"]
            ).max(),
        )
        result_df = pd.concat(result_df_list).reset_index(level=0)
        result_df.to_csv("num_data.csv")
        tqdm.write("=====" * 15)

    result_df = pd.concat(result_df_list).reset_index(level=0)
    result_df.to_csv("num_data.csv")

    # plot(vary_list=num_data_list, result_df=result_df, variable_name="num_data")
    # plot_normalize(vary_list=num_data_list, result_df=result_df, variable_name="num_data")


if __name__ == "__main__":
    main()
