#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
from matplotlib import rcParams

matplotlib.use("TkAgg")

# 改进后的颜色列表（16色，补全逗号并合法化）
color_list = [
    "#C99E8C",  # 脏橘
    "#465E65",  # 冷蓝
    "#FE4567",  # 山茶红
    "#57C3C2",  # 石绿
    "#BCF60B",  # 酸绿
    "#53589A",  # 靛紫
    "#C1C1C1",  # 灰
    "#76905C",  # 橄榄绿
    "#FFCF79",  # 柔黄
    "#6C60AB",  # 蓝紫
    "#C7D1EA",  # 云灰
    "#815DAB",  # 浅紫
    "#EF7C29",  # 橙
    "#8187C5",  # 雾蓝
    "#D56C9B",  # 墨粉
    "#82CFFD",  # 浅蓝
]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Euclid"     # Default: "font.family": "Helvetica"
})

def plot_cost_hist(cost_list, bins=25, color='#57C3C2', is_average=True,
                   title=r"$\mathbf{Cost\ Distribution}$",
                   xlabel=r"$\mathbf{Cost}$",
                   ylabel=r"$\mathbf{Probability}$"):
    plt.figure()
    cost_list.sort(key=lambda x: x[0])
    data_list = [c[0] / c[2] if is_average else c[0] for c in cost_list]

    sns.histplot(data_list, bins=bins, kde=True, color=color, edgecolor=color, stat="probability", alpha=0.6)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True)
    plt.tight_layout()


def plot_cost_hists_multi(*cost_lists, bins=25, colors=None, labels=None,
                          is_average=True,
                          title=r"$\mathbf{Cost\ Distribution}$",
                          xlabel=r"$\mathbf{Cost}$",
                          ylabel=r"$\mathbf{Probability}$"):
    plt.figure()

    def extract_data(cost_list):
        cost_list.sort(key=lambda x: x[0])
        return [item[0] / item[2] if is_average else item[0] for item in cost_list]

    num_datasets = len(cost_lists)

    if colors is None:
        colors = color_list[:num_datasets]
        #colors = sns.color_palette(n_colors=num_datasets)
    if labels is None:
        labels = [f'Dataset {i + 1}' for i in range(num_datasets)]

    '''
        ChatGPT said:
        
        我看你这张图，柱子（histogram）本身最高在 1 以下，但那条红色的光滑曲线（KDE）峰值超过了 1。
        
        这是合理的，因为：
        
            你用的是 stat="probability"，所以直方图柱子的高度是概率，不会超过 1。
        
            但是 kde=True 画出来的曲线是概率密度函数（PDF），它的纵轴是密度，不是概率，完全可能超过 1（尤其是数据集中在一个很窄的范围时，密度峰值会很高）。
        
            你的纵轴标签是 Probability，但它同时在表示直方图的概率和 KDE 的密度，这会让人误解。
        
        如果想避免混淆，你有两个选择：
        
            分开画：直方图用概率（stat="probability"），KDE 单独画在另一个 y 轴（密度轴）。
        
            统一成密度：把直方图也改成 stat="density"，然后 y 轴标签改为 Probability Density。这样曲线和柱子单位一致，高度超过 1 就正常了。    
    '''
    for i, cost_list in enumerate(cost_lists):
        data = extract_data(cost_list)
        sns.histplot(data, bins=bins, kde=True, color=colors[i], edgecolor=colors[i],
                     label=labels[i], stat="probability", alpha=0.6)

    # 标题 & 标签字体
    plt.title(title, fontsize=28)
    plt.xlabel(xlabel, fontsize=26)
    plt.ylabel(ylabel, fontsize=26)

    # 坐标刻度字体
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # 图例字体
    plt.legend(fontsize=20)

    # 网格线加粗
    plt.grid(True, linewidth=1.2)
    plt.tight_layout()


def plot_cost_hists_together_4_comparision(cost_groups, bins=25,
                                           colors_pi=None, colors_gamma=None,
                                           labels_pi=None, labels_gamma=None,
                                           is_average=True,
                                           title=r"$\mathbf{Cost\ Distribution}$",
                                           xlabel=r"$\mathbf{Cost}$",
                                           ylabel=r"$\mathbf{Probability}$"):

    plt.figure()

    def extract_data(cost_list):
        cost_list.sort(key=lambda x: x[0])
        return [item[0] / item[2] if is_average else item[0] for item in cost_list]

    num_groups = len(cost_groups)

    if colors_pi is None:
        colors_pi = color_list[:num_groups]
    if colors_gamma is None:
        colors_gamma = color_list[8:8 + num_groups]
    if labels_pi is None:
        labels_pi = [f"$\\pi_{{{i+1}}}$" for i in range(num_groups)]
    if labels_gamma is None:
        labels_gamma = [f"$\\gamma_{{{i+1}}}$" for i in range(num_groups)]

    for i in range(num_groups):
        group = cost_groups[i]
        if len(group) != 2:
            raise ValueError(f"Group {i} does not contain exactly 2 cost lists (π and γ).")

        cost_pi, cost_gamma = group
        data_pi = extract_data(cost_pi)
        data_gamma = extract_data(cost_gamma)

        # 画pi（透明度稍低），再画 gamma（图像在上层）
        sns.histplot(data_pi, bins=bins, kde=True, color=colors_pi[i], edgecolor=colors_pi[i],
                     label=labels_pi[i], stat="probability", alpha=0.4, linewidth=1.5)

        sns.histplot(data_gamma, bins=bins, kde=True, color=colors_gamma[i], edgecolor=colors_gamma[i],
                     label=labels_gamma[i], stat="probability", alpha=0.6, linewidth=1.5)

    #
    # 标题 & 标签字体
    plt.title(title, fontsize=36)
    plt.xlabel(xlabel, fontsize=34)
    plt.ylabel(ylabel, fontsize=34)

    # 坐标刻度字体
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

    # 图例字体
    plt.legend(loc='best', fontsize=28)

    # 网格线加粗
    plt.grid(True, linewidth=1.5)
    plt.tight_layout()

def plot_cost_hists_together_4_comparision_multi_groups(
        cost_groups, bins=25,
        colors_pi=None, colors_gamma=None,
        labels_pi=None, labels_gamma=None,
        is_average=True,
        titles=None,
        xlabel=r"$\mathbf{Cost}$",
        ylabel=r"$\mathbf{Probability}$"):
    """
    cost_groups: List of 2x2 groups, e.g. [
        [[cost_pi1, cost_gamma1], [cost_pi2, cost_gamma2]],
        ...
    ]

    在一张图内用多个subplot显示多组比较结果。

    Usage:
        plot_cost_hists_together_4_comparision_multi_groups(
        cost_groups=[
            [[cost_pi1, cost_gamma1], [cost_pi2, cost_gamma2]],
            [[cost_pi3, cost_gamma3], [cost_pi4, cost_gamma4]]
        ],
        labels_pi=[["$\pi_1$", "$\pi_2$"], ["$\pi_3$", "$\pi_4$"]],
        labels_gamma=[["$\gamma_1$", "$\gamma_2$"], ["$\gamma_3$", "$\gamma_4$"]],
        titles=[r"$\mathbf{Mission\ 1}$", r"$\mathbf{Mission\ 2}$"]
    )
    """

    def extract_data(cost_list):
        cost_list.sort(key=lambda x: x[0])
        return [item[0] / item[2] if is_average else item[0] for item in cost_list]

    num_groups = len(cost_groups)
    if colors_pi is None:
        colors_pi = color_list[:2 * num_groups]
        # colors_pi = ["#C99E8C", "#57C3C2"] * num_groups
    if colors_gamma is None:
        colors_gamma = color_list[8:8 + 2 * num_groups]
        # colors_gamma = ["#465E65", "#FE4567"] * num_groups
    if labels_pi is None:
        labels_pi = [[f"$\\pi_{{{2 * i + 1}}}$", f"$\\pi_{{{2 * i + 2}}}$"] for i in range(num_groups)]
    if labels_gamma is None:
        labels_gamma = [[f"$\\gamma_{{{2 * i + 1}}}$", f"$\\gamma_{{{2 * i + 2}}}$"] for i in range(num_groups)]
    if titles is None:
        titles = [f"$\\mathbf{{Group\\ {i + 1}}}$" for i in range(num_groups)]

    fig, axs = plt.subplots(nrows=1, ncols=num_groups, figsize=(7 * num_groups, 5), squeeze=False)

    handles_all = []
    labels_all = []

    for i, group in enumerate(cost_groups):
        if len(group) != 2 or any(len(pair) != 2 for pair in group):
            raise ValueError(f"Group {i} must be 2x2: [[pi1, gamma1], [pi2, gamma2]]")

        (cost_pi1, cost_gamma1), (cost_pi2, cost_gamma2) = group
        data_pi1 = extract_data(cost_pi1)
        data_gamma1 = extract_data(cost_gamma1)
        data_pi2 = extract_data(cost_pi2)
        data_gamma2 = extract_data(cost_gamma2)

        ax = axs[0][i]

        h1 = sns.histplot(data_pi1, bins=bins, kde=True, color=colors_pi[2 * i], edgecolor=colors_pi[2 * i],
                          label=labels_pi[i][0], stat="probability", alpha=0.4, ax=ax)
        h2 = sns.histplot(data_gamma1, bins=bins, kde=True, color=colors_gamma[2 * i], edgecolor=colors_gamma[2 * i],
                          label=labels_gamma[i][0], stat="probability", alpha=0.6, ax=ax)
        h3 = sns.histplot(data_pi2, bins=bins, kde=True, color=colors_pi[2 * i + 1], edgecolor=colors_pi[2 * i + 1],
                          label=labels_pi[i][1], stat="probability", alpha=0.4, ax=ax)
        h4 = sns.histplot(data_gamma2, bins=bins, kde=True, color=colors_gamma[2 * i + 1],
                          edgecolor=colors_gamma[2 * i + 1],
                          label=labels_gamma[i][1], stat="probability", alpha=0.6, ax=ax)

        # 收集 legend handle 和 label
        for h in ax.get_legend_handles_labels()[0]:
            handles_all.append(h)
        for l in ax.get_legend_handles_labels()[1]:
            labels_all.append(l)

        ax.set_title(titles[i], fontsize=18)
        ax.set_title(titles[i], fontsize=18)
        ax.set_xlabel("")  # 清除每个子图单独的 x-label
        ax.set_ylabel("")  # 清除每个子图单独的 y-label
        if i != 0:
            ax.tick_params(left=False, labelleft=False)
        ax.grid(True)
        ax.legend().remove()  # 移除单独图例

    # 设置统一 legend
    fig.legend(handles_all, labels_all, loc='center right', fontsize=14, frameon=False)

    # 添加统一 x/y label
    fig.text(0.5, 0.04, xlabel, ha='center', fontsize=18)
    fig.text(0.06, 0.5, ylabel, va='center', rotation='vertical', fontsize=18)

    # 调整 layout，留出下方 x-label、左侧 y-label、右侧 legend 空间
    plt.tight_layout(rect=[0.1, 0.08, 0.88, 1])
