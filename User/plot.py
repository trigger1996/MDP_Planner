import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 使用 LaTeX 字体（需本地安装 LaTeX 发行版）
plt.rcParams.update({
    "text.usetex": False,  # 不使用外部 LaTeX
    "font.family": "serif",
    "font.serif": ["Computer Modern", "Times New Roman"],
    "mathtext.fontset": "cm",  # 使用 LaTeX 的 Computer Modern 数学字体
    "axes.unicode_minus": False,
})

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
