#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
from matplotlib import rcParams

# === 手动加载 Euclid 字体 ===
font_dir = os.path.expanduser("~/.fonts")

# 选定字体文件（根据你列出的文件）
euclid_regular_path = os.path.join(font_dir, "Euclid Extra Regular.ttf")
euclid_italic_path = os.path.join(font_dir, "Euclid Italic.ttf")
euclid_bold_path = os.path.join(font_dir, "Euclid Extra Bold.ttf")

# 加载 FontProperties
font_prop_regular = fm.FontProperties(family=["Euclid Math One Regular", "Euclid Math Two Regular", "Euclid Symbol", "Euclid"])
font_prop_italic  = fm.FontProperties(family=["Euclid Symbol", "Euclid Italic", "Euclid"], style="italic")
font_prop_bold    = fm.FontProperties(fname="Euclid")


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

def set_plot_fonts():

    font_prop_regular_t = fm.FontProperties(fname=euclid_regular_path)
    font_prop_italic_t = fm.FontProperties(fname=euclid_italic_path)
    font_prop_bold_t = fm.FontProperties(fname=euclid_bold_path)

    # 提取 font name（用于 rcParams）
    name_regular = font_prop_regular_t.get_name()  # 如：'Euclid Extra Regular'
    name_italic  = font_prop_italic_t.get_name()  # 如：'Euclid Italic'
    name_bold    = font_prop_bold_t.get_name()  # 如：'Euclid Extra Bold'

    # 设置 matplotlib 字体
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": name_regular,
        "axes.unicode_minus": False,

        # 设置 mathtext 使用 Euclid 家族
        "mathtext.fontset": "custom",
        "mathtext.rm": name_regular,  # 普通 math
        "mathtext.it": name_italic,  # italic
        "mathtext.bf": name_bold,  # bold
        "mathtext.sf": name_regular,
    })
    # plt.rcParams.update({
    #     "text.usetex": False,
    #     "font.family": "Euclid",  # 强制字体名
    #     "mathtext.fontset": "custom",  # 使用 LaTeX 的 Computer Modern 数学字体, default: "mathtext.fontset": "cm",
    #     "mathtext.rm": "Euclid",
    #     "mathtext.it": "Euclid",
    #     "mathtext.bf": "Euclid",
    #     "axes.unicode_minus": False,
    # })


def plot_cost_hist(cost_list, bins=25, color='#57C3C2', is_average=True,
                   title=r"$\mathbf{Cost\ Distribution}$",
                   xlabel=r"$\mathbf{Cost}$",
                   ylabel=r"$\mathbf{Probability}$"):
    plt.figure()
    cost_list.sort(key=lambda x: x[0])
    data_list = [c[0] / c[2] if is_average else c[0] for c in cost_list]

    sns.histplot(data_list, bins=bins, kde=True, color=color, stat="probability", alpha=0.6)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
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

    for i, cost_list in enumerate(cost_lists):
        data = extract_data(cost_list)
        sns.histplot(data, bins=bins, kde=True, color=colors[i],
                     label=labels[i], stat="probability", alpha=0.6)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_cost_hists_together_4_comparision(cost_groups, bins=25,
                                           colors_pi=None, colors_gamma=None,
                                           labels_pi=None, labels_gamma=None,
                                           is_average=True,
                                           title=r"$\mathbf{Cost\ Distribution}$",
                                           xlabel=r"$\mathbf{Cost}$",
                                           ylabel=r"$\mathbf{Probability}$"):
    set_plot_fonts()

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
        sns.histplot(data_pi, bins=bins, kde=True, color=colors_pi[i],
                     label=labels_pi[i], stat="probability", alpha=0.4, linewidth=1.5)

        sns.histplot(data_gamma, bins=bins, kde=True, color=colors_gamma[i],
                     label=labels_gamma[i], stat="probability", alpha=0.6, linewidth=1.5)

    #
    plt.title(title,   fontproperties=font_prop_regular, fontsize=14)
    plt.xlabel(xlabel, fontproperties=font_prop_italic,  fontsize=12)
    plt.ylabel(ylabel, fontproperties=font_prop_italic,  fontsize=12)
    plt.legend(loc='best', prop=font_prop_italic)  # 注意这里才是 prop
    plt.grid(True)
    plt.tight_layout()

def plot_cost_hists_together_4_comparision_in_one_slide():
    # TODO
    set_plot_fonts()