import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from matplotlib import colors
import matplotlib.patches as pc

def print_c(data, color=32):
    """
    https://blog.csdn.net/XianZhe_/article/details/113075983
    颜色样式打印输出功能
    :param data: 打印内容
    :param color: 指定颜色, 默认为绿色(32)
    :return:
    """
    if isinstance(color, int):
        color = str(color)
    print(f"\033[1;{color}m{data}\033[0m")

def visualiza_in_animation(mdp, s0, XX, LL, UU, MM, name=None):
    grid_size = 0.25

    fig0 = plt.figure()
    ax = fig0.add_subplot(1, 1, 1)

    x = np.linspace(0, 2 * np.pi, 5000)
    y = np.exp(-x) * np.cos(2 * np.pi * x)
    line, = ax.plot(x, y, color="cornflowerblue", lw=3)
    ax.set_ylim(-1.1, 1.1)

    # 清空当前帧
    def init():
        line.set_ydata([np.nan] * len(x))
        return line,

    # 更新新一帧的数据
    def update(frame):
        line.set_ydata(np.exp(-x) * np.cos(2 * np.pi * x + float(frame) / 100))
        return line,

    # 调用 FuncAnimation
    ani = FuncAnimation(fig0
                        , update
                        , init_func=init
                        , frames=200
                        , interval=2
                        , blit=True
                        )
    #
    # https://blog.csdn.net/qq_42727752/article/details/117339892

    # step 1 find all grids from motion mdp
    grids = []
    ap_list = []
    for state_t in list(mdp.nodes.keys()):
        #
        x_t = state_t[0]
        y_t = state_t[1]
        #
        label_t = []
        act_t = []
        for key_t in mdp.nodes[state_t]:
            if key_t == 'label':
                label_t = list(mdp.nodes[state_t]['label'].keys())[0]         # 这个是试出来的
                label_t = tuple(label_t)
                #
                for label_t_t in label_t:
                    ap_list.append(label_t_t)
            if key_t == 'act':
                act_t = list(mdp.nodes[state_t]['act'])
        #
        grids.append((x_t, y_t, label_t))
    grids = list(set(grids))
    ap_list = list(set(ap_list))

    startxy = [s0[0], s0[1]]


    goalxy = [2, 0.5]
    obsxy = [[1, 0], [1, 0.5], [1, 1]]

    Ys = max(x_y_ap[1] for x_y_ap in grids)
    Xs = max(x_y_ap[0] for x_y_ap in grids)
    Ym = min(x_y_ap[1] for x_y_ap in grids)
    Xm = min(x_y_ap[0] for x_y_ap in grids)


    def xy2sub(len, x, y, grid_size):
        r = int(len - y / grid_size - 1)
        c = int(x  / grid_size)
        return [r, c]

    def sub2xy(len, r, c, grid_size):
        x = c * grid_size
        y = len - r * grid_size - 1
        return [x, y]

    # 其中X和矩阵地图的cols对应
    rows = int(Ys / grid_size)
    cols = int(Xs / grid_size)
    # 创建全部为空地的地图栅格，其中空地以数字1表征
    # ！！！注意ones(行列个数，因此rows需要+1)
    field = np.ones([rows, cols])

    # 修改栅格地图中起始点和终点的数值，其中起点以数值4表征，终点以数值5表示
    startsub = xy2sub(rows, startxy[0], startxy[1], grid_size)
    goalsub = xy2sub(rows, goalxy[0], goalxy[1], grid_size)
    field[startsub[0], startsub[1]] = 4
    field[goalsub[0], goalsub[1]] = 5

    # 修改栅格地图中障碍物的数值，其中以数值5表示
    for i in range(len(obsxy)):
        obssub = xy2sub(rows, obsxy[i][0], obsxy[i][1], grid_size)
        field[obssub[0], obssub[1]] = 2

    # 设置画布属性
    #plt.figure(figsize=(cols, rows))
    plt.figure()
    plt.xlim(Ym - grid_size, Ys + grid_size)
    plt.ylim(Xm - grid_size, Xs + grid_size)
    plt.xticks(np.arange(Xs))
    plt.yticks(np.arange(Ys))

    # 绘制障碍物XY位置
    for i in range(len(obsxy)):
        plt.gca().add_patch(pc.Rectangle((obsxy[i][0] - grid_size / 2, obsxy[i][1] - grid_size / 2), grid_size, grid_size, color='k'))

    # 绘制起点，终点
    plt.gca().add_patch(pc.Rectangle((startxy[0] - grid_size / 2, startxy[1] - grid_size / 2), grid_size, grid_size, color='yellow'))
    plt.gca().add_patch(pc.Rectangle((goalxy[0] - grid_size / 2, goalxy[1] - grid_size / 2), grid_size, grid_size, color='m'))

    return field












def visualize_run_sequence(XX, LL, UU, MM, name=None, is_visuaize=False):
    # -----plot the sequence of states for the test run
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    N = len(XX)
    # print 'N: %s' %N
    # ----
    for n in range(0, N):
        X = list(XX[n])
        L = list(LL[n])
        U = list(UU[n])
        M = list(MM[n])
        K = len(X)
        # print 'K: %s' %K
        RAD = 0.3
        for k in range(0, K):
            if M[k] == 0:
                color = 'blue'
            if M[k] == 1:
                color = 'magenta'
            if M[k] == 2:
                color = 'black'
            # ----
            rec = matplotlib.patches.Rectangle((4*k-RAD, 3*n-RAD),
                                               RAD*2, RAD*2,
                                               fill=False,
                                               edgecolor=color,
                                               linewidth=3,
                                               ls='solid',
                                               alpha=1)
            ax.add_patch(rec)
            setstr = r''
            for s in L[k]:
                setstr += s
                setstr += ','
            ax.text(4*k-RAD, 3*n+RAD*4, r'$(%s, \{%s\})$' %
                    (str(X[k]), str(setstr)), fontsize=6, fontweight='bold')
            # ----
            if (k <= K-2):
                line = matplotlib.lines.Line2D([4*k+RAD, 4*k+4-RAD],
                                               [3*n, 3*n],
                                               linestyle='-',
                                               linewidth=1,
                                               color='black')
                ax.add_line(line)
                actstr = r''
                for s in U[k]:
                    actstr += s
                ax.text(4*k+2, 3*n+RAD, r'%s' %
                        str(actstr), fontsize=6, fontweight='bold')
    ax.set_aspect(0.7)
    ax.set_xlim(-1, 4*K)
    ax.set_ylim(-1, 3*N)
    ax.set_xlabel(r'$state\; sequence$')
    ax.set_ylabel(r'$run$')
    # ax.axis('off')
    # if name:
    #     plt.savefig('%s.pdf' % name, bbox_inches='tight')

    if is_visuaize:
        plt.show()

    return figure
