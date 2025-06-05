import matplotlib.pyplot as plt
import seaborn as sns

def plot_cost_hist(cost_list, bins=25, color='g', is_average=True, title="Cost Distribution", xlabel="Cost", ylabel="Probability"):
    plt.figure()
    cost_list.sort(key=lambda x: x[0])
    data_list = []
    for i in range(0, cost_list.__len__()):
        if is_average:
            data_list.append(cost_list[i][0] / cost_list[i][2])
        else:
            data_list.append(cost_list[i][0])

    sns.histplot(data_list, bins=bins, kde=True, color=color, stat="probability")    # stat="density" "probability"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_cost_hists_multi(*cost_lists, bins=25, colors=None, labels=None,
                          is_average=True, title="Cost Distribution",
                          xlabel="Cost", ylabel="Probability"):
    plt.figure()

    def extract_data(cost_list):
        cost_list.sort(key=lambda x: x[0])
        return [item[0] / item[2] if is_average else item[0] for item in cost_list]

    num_datasets = len(cost_lists)

    # 设置默认颜色和标签
    if colors is None:
        colors = sns.color_palette(n_colors=num_datasets)
    if labels is None:
        labels = [f'Dataset {i + 1}' for i in range(num_datasets)]

    for i, cost_list in enumerate(cost_lists):
        data = extract_data(cost_list)
        sns.histplot(data, bins=bins, kde=True, color=colors[i], label=labels[i],
                     stat="probability", alpha=0.5)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()