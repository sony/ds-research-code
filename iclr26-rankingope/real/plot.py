# Copyright (c) 2026 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from tqdm import tqdm


def plot(vary_list, result_df, variable_name):

    registered_colors = {
        "IPS": "tab:red",
        "IIPS": "tab:blue",
        "RIPS": "tab:purple",
        "CIPS": "tab:green",
        "CDR": "tab:orange",
        "CIPS (estimate)": "tab:gray",
        "CDR (estimate)": "tab:brown",
    }

    legend = ["IPS", "IIPS", "RIPS", "CIPS", "CDR", "CIPS (estimate)", "CDR (estimate)"]
    palette = [registered_colors[est] for est in legend]

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(30, 7), tight_layout=True)

    # MSE
    ax_mse = fig.add_subplot(1, 3, 1)

    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="se",
        hue="est",
        ax=ax_mse,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        marker="o",
        errorbar=None,
    )

    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="se",
        hue="est",
        ax=ax_mse,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        marker="o",
    )
    # yaxis
    ax_mse.set_yscale("log")
    ax_mse.set_ylabel("")
    ax_mse.tick_params(axis="y", labelsize=25)
    ax_mse.yaxis.set_label_coords(-0.1, 0.5)
    # xaxis
    if variable_name == "num_data":
        ax_mse.set_xscale("log")
    ax_mse.set_xlabel(f"{variable_name}", fontsize=30)
    ax_mse.tick_params(axis="x", labelsize=25)
    ax_mse.set_xticks(vary_list)
    ax_mse.set_xticklabels(vary_list, fontsize=25)
    ax_mse.xaxis.set_label_coords(0.5, -0.1)
    plt.title("MSE", fontsize=35)

    # Bias
    ax_bias = fig.add_subplot(1, 3, 2)
    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="bias",
        hue="est",
        ax=ax_bias,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        errorbar=None,
        marker="o",
    )

    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="bias",
        hue="est",
        ax=ax_bias,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        marker="o",
    )
    # yaxis
    ax_bias.set_yscale("log")
    ax_bias.set_ylabel("")
    ax_bias.tick_params(axis="y", labelsize=25)
    ax_bias.yaxis.set_label_coords(-0.1, 0.5)
    # xaxis
    if variable_name == "num_data":
        ax_bias.set_xscale("log")
    ax_bias.set_xlabel(f"{variable_name}", fontsize=30)
    ax_bias.tick_params(axis="x", labelsize=25)
    ax_bias.set_xticks(vary_list)
    ax_bias.set_xticklabels(vary_list, fontsize=25)
    ax_bias.xaxis.set_label_coords(0.5, -0.1)
    plt.title("Squared Bias", fontsize=35)

    # Variance
    ax_variance = fig.add_subplot(1, 3, 3)

    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="variance",
        hue="est",
        ax=ax_variance,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        marker="o",
        errorbar=None,
    )

    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="variance",
        hue="est",
        ax=ax_variance,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        marker="o",
        # errorbar=None,
    )
    # yaxis
    ax_variance.set_yscale("log")
    ax_variance.set_ylabel("")
    ax_variance.tick_params(axis="y", labelsize=25)
    ax_variance.yaxis.set_label_coords(0.5, -0.1)
    # xaxis
    if variable_name == "num_data":
        ax_variance.set_xscale("log")
    ax_variance.set_xlabel(f"{variable_name}", fontsize=30)
    ax_variance.tick_params(axis="x", labelsize=25)
    ax_variance.set_xticks(vary_list)
    ax_variance.set_xticklabels(vary_list, fontsize=25)
    ax_variance.xaxis.set_label_coords(0.5, -0.1)

    plt.title("Variance", fontsize=35)

    # fig.legend(
    #     legend, fontsize=25,
    #     bbox_to_anchor=(0.5, 1.05),
    #     ncol=len(legend), loc='center',
    # )
    plt.legend(legend, fontsize=20)
    plt.savefig("test.png")
    # plt.show()


def plot_normalize(vary_list, result_df, variable_name):

    pi_e_value = np.abs(result_df["pi_e_value"])
    result_df["se"] /= pi_e_value
    result_df["bias"] /= pi_e_value
    result_df["variance"] /= pi_e_value

    registered_colors = {
        "IPS": "tab:red",
        "IIPS": "tab:blue",
        "RIPS": "tab:purple",
        "CIPS": "tab:green",
        "CDR": "tab:orange",
        "CIPS (estimate)": "tab:gray",
        "CDR (estimate)": "tab:brown",
    }

    legend = ["IPS", "IIPS", "RIPS", "CIPS", "CDR", "CIPS (estimate)", "CDR (estimate)"]
    palette = [registered_colors[est] for est in legend]

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(30, 7), tight_layout=True)

    # MSE
    ax_mse = fig.add_subplot(1, 3, 1)

    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="se",
        hue="est",
        ax=ax_mse,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        marker="o",
        errorbar=None,
    )

    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="se",
        hue="est",
        ax=ax_mse,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        marker="o",
    )
    # yaxis
    # ax_mse.set_yscale("log")
    ax_mse.set_ylabel("")
    ax_mse.tick_params(axis="y", labelsize=25)
    ax_mse.yaxis.set_label_coords(-0.1, 0.5)
    # xaxis
    if variable_name == "num_data":
        ax_mse.set_xscale("log")
    ax_mse.set_xlabel(f"{variable_name}", fontsize=30)
    ax_mse.tick_params(axis="x", labelsize=25)
    ax_mse.set_xticks(vary_list)
    ax_mse.set_xticklabels(vary_list, fontsize=25)
    ax_mse.xaxis.set_label_coords(0.5, -0.1)
    plt.title("Normalized MSE", fontsize=35)

    # Bias
    ax_bias = fig.add_subplot(1, 3, 2)
    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="bias",
        hue="est",
        ax=ax_bias,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        errorbar=None,
        marker="o",
    )

    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="bias",
        hue="est",
        ax=ax_bias,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        marker="o",
    )
    # yaxis
    # ax_bias.set_yscale("log")
    ax_bias.set_ylabel("")
    ax_bias.tick_params(axis="y", labelsize=25)
    ax_bias.yaxis.set_label_coords(-0.1, 0.5)
    # xaxis
    if variable_name == "num_data":
        ax_bias.set_xscale("log")
    ax_bias.set_xlabel(f"{variable_name}", fontsize=30)
    ax_bias.tick_params(axis="x", labelsize=25)
    ax_bias.set_xticks(vary_list)
    ax_bias.set_xticklabels(vary_list, fontsize=25)
    ax_bias.xaxis.set_label_coords(0.5, -0.1)
    plt.title("Squared Bias", fontsize=35)

    # Variance
    ax_variance = fig.add_subplot(1, 3, 3)

    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="variance",
        hue="est",
        ax=ax_variance,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        marker="o",
        errorbar=None,
    )

    sns.lineplot(
        linewidth=4,
        legend=False,
        # style="est",
        x=variable_name,
        y="variance",
        hue="est",
        ax=ax_variance,
        data=result_df,
        markers=True,
        dashes=False,
        markersize=15,
        palette=palette,
        marker="o",
        # errorbar=None,
    )
    # yaxis
    # ax_variance.set_yscale("log")
    ax_variance.set_ylabel("")
    ax_variance.tick_params(axis="y", labelsize=25)
    ax_variance.yaxis.set_label_coords(0.5, -0.1)
    # xaxis
    if variable_name == "num_data":
        ax_variance.set_xscale("log")
    ax_variance.set_xlabel(f"{variable_name}", fontsize=30)
    ax_variance.tick_params(axis="x", labelsize=25)
    ax_variance.set_xticks(vary_list)
    ax_variance.set_xticklabels(vary_list, fontsize=25)
    ax_variance.xaxis.set_label_coords(0.5, -0.1)

    plt.title("Variance", fontsize=35)

    # fig.legend(
    #     legend, fontsize=25,
    #     bbox_to_anchor=(0.5, 1.05),
    #     ncol=len(legend), loc='center',
    # )
    plt.legend(legend, fontsize=20)
    plt.savefig("test_normalize.png")
    # plt.show()
