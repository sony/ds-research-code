# Copyright (c) 2025 Sony Group Corporation and Hanjuku-kaso Co., Ltd. All Rights Reserved.
#
# This software is released under the MIT License.


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

plt.style.use("ggplot")

import conf


def show_loss(opfv_opl, ips, dr, reg):
    num_iter = len(opfv_opl.train_loss)
    xticks = range(1, num_iter + 1)
    plt.style.use("ggplot")
    plt.plot(xticks, ips.train_loss, label="IPS")
    plt.plot(xticks, dr.train_loss, label="DR")
    plt.plot(xticks, reg.train_loss, label="Regression-based")
    plt.plot(xticks, opfv_opl.train_loss, label="OPFV")
    plt.xlabel("epochs")
    plt.xticks(xticks)
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()
    plt.show()


def show_value_train(opfv_opl, ips, dr, reg, prog_DM):
    plt.style.use("ggplot")
    plt.plot(ips.train_value, label="IPS train")
    plt.plot(dr.train_value, label="DR train")
    plt.plot(reg.train_value, label="REG train")
    plt.plot(prog_DM.train_value, label="Prog. train")
    plt.plot(opfv_opl.train_value, label="OPFV train")
    plt.xlabel("epochs")
    plt.ylabel("value")
    plt.title("Train policy Value")
    plt.legend()
    plt.show()


def show_value_test(opfv_opl, ips, dr, reg, prog_DM):
    plt.style.use("ggplot")
    plt.plot(ips.test_value, label="IPS test")
    plt.plot(dr.test_value, label="DR test")
    plt.plot(reg.test_value, label="REG test")
    plt.plot(prog_DM.test_value, label="Prog. test")
    plt.plot(opfv_opl.test_value, label="OPFV test")
    plt.xlabel("epochs")
    plt.ylabel("value")
    plt.title("Test Policy Value")
    plt.legend()
    plt.show()


plt.style.use("ggplot")
registered_colors = {}
legend = []

if conf.flag_include_behavior_policy == True:
    registered_colors["Bahavior"] = "tab:pink"
    legend.append("Bahavior")

if conf.flag_include_best_policy == True:
    registered_colors["Best"] = "tab:orange"
    legend.append("Best")

if conf.flag_include_RegBased == True:
    registered_colors["RegBased"] = "tab:purple"
    legend.append("RegBased")

if conf.flag_include_IPS_PG == True:
    registered_colors["IPS-PG"] = "tab:red"
    legend.append("IPS-PG")

if conf.flag_include_DR_PG == True:
    registered_colors["DR-PG"] = "tab:blue"
    legend.append("DR-PG")

if conf.flag_include_Prognosticator == True:
    registered_colors["Prognosticator"] = "tab:grey"
    legend.append("Prognosticator")

registered_colors["OPFV-PG"] = "tab:green"
legend.append("OPFV-PG")


num_estimators = len(legend)

palette = [registered_colors[est] for est in legend]

line_legend_elements = [
    Line2D(
        [0],
        [0],
        color=registered_colors[est],
        linewidth=5,
        marker="o",
        markerfacecolor=registered_colors[est],
        markersize=10,
        label=est,
    )
    for est in legend
]


def plot_result(
    result_df,
    x,
    xlabel,
    xticklabels,
):
    fig, ax = plt.subplots(figsize=(15, 8), tight_layout=True)
    sns.lineplot(
        linewidth=7,
        marker="o",
        markersize=15,
        markers=True,
        x=x,
        y="value",
        hue="method",
        ax=ax,
        palette=palette,
        legend=False,
        data=result_df,
    )
    # yaxis
    ax.set_ylabel("Trained Policy Value", fontsize=25)
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    # xaxis
    if x in ["n_trains"]:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(f"{xlabel}", fontsize=26)
    ax.set_xticks(xticklabels)
    ax.set_xticklabels(xticklabels, fontsize=18)

    fig.legend(
        handles=line_legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=num_estimators,
        fontsize=25,
    )
