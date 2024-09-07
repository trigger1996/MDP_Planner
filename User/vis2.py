import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib import colors
import matplotlib.patches as pc
from matplotlib.style import available

color_set = np.array([[188, 195, 48],                # in rgb
                      [204, 0,   111],
                      [97,  58,  139],
                      [240, 130, 0],
                      [126, 79, 142],
                      [0,   97,  147],
                      [241, 180, 100],
                      [50,  128, 98],
                      [232, 181, 160],
                      [235, 152, 80],
                      [231, 228, 95],
                      [35, 173, 229],
                      [250, 204 ,126],
                      [192, 207, 91],
                      [116, 190, 215],
                      [31,  174, 103],
                      [79,  138, 201],
                      [244, 235, 113],
                      [187, 164, 194],
                      [216, 228, 131],
                      [148, 197, 219],
                      [0,    98, 150],
                      [167,  87, 102]])
color_set = np.multiply(color_set, 1. / 256.)

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


def draw_fuselage(ax, center, heading, face_color='lightblue', edge_color='darkblue', alpha=1.):
    yaw = 0
    if type(heading) == int or type(heading) == float:
        yaw = heading
    elif type(heading) == str:
        if heading == 'N':
            yaw = 0
        elif heading == 'E':
            yaw = math.pi / 2
        elif heading == 'S':
            yaw = math.pi
        elif heading == 'W':
            yaw = -math.pi / 2

    # 五边形的顶点坐标

    vertices = np.array([
        (0.03,   0.0),
        (0,      0.05),
        (-0.05,  0.05),
        (-0.05, -0.05),
        (0,     -0.05)
    ])                          # (y, x)

    for i in range(0, vertices.shape[0]):
        x_t = vertices[i,1]
        y_t = vertices[i,0]
        vertices[i, 0] = x_t * -math.sin(yaw) + y_t * math.cos(yaw) + center[1]         # y
        vertices[i, 1] = x_t *  math.cos(yaw) + y_t * math.sin(yaw) + center[0]         # x

    # 创建五边形
    polygon = Polygon(vertices, closed=True, edgecolor=edge_color, facecolor=face_color, alpha=alpha)
    ax.add_patch(polygon)

    return ax, polygon

def visualize_trajectories(mdp, s0, XX, LL, UU, MM, name=None, is_gradient_color=True):
    #
    # XX: 状态序列, list(list()), 一次给了多个run
    # LL: X -> 2^AP 命题序列
    # UU: 控制序列
    # MM: AP是否满足原子命题的序列
    #
    # https://blog.csdn.net/qq_42727752/article/details/117339892
    def state_2_grid_center(state):
        x_t = state[0]
        y_t = state[1]
        return x_t, y_t

    def xy2sub(len, x, y, grid_size):
        r = int(len - y / grid_size - 1)
        c = int(x  / grid_size)
        return [r, c]

    def sub2xy(len, r, c, grid_size):
        x = c * grid_size
        y = len - r * grid_size - 1
        return [x, y]

    # step 1 find all grids from motion mdp
    grid_size = 0.5
    grids = []
    ap_list = []
    for state_t in list(mdp.nodes.keys()):
        #
        #x_t = state_t[0]
        #y_t = state_t[1]
        x_t, y_t = state_2_grid_center(state_t)
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
    ap_list.sort()

    Ys = max(x_y_ap[1] for x_y_ap in grids)
    Xs = max(x_y_ap[0] for x_y_ap in grids)
    Ym = min(x_y_ap[1] for x_y_ap in grids)
    Xm = min(x_y_ap[0] for x_y_ap in grids)


    # 其中X和矩阵地图的cols对应
    rows = int(Ys / grid_size)
    cols = int(Xs / grid_size)
    # 创建全部为空地的地图栅格，其中空地以数字1表征
    # ！！！注意ones(行列个数，因此rows需要+1)
    field = np.ones([rows, cols])

    run_xy_yaw_set = []
    for run_t in XX:
        run_xy_t = []
        for state_t in run_t:
            x_t, y_t = state_2_grid_center(state_t)
            heading_t = state_t[2]
            run_xy_t.append((x_t, y_t, heading_t,))
        run_xy_yaw_set.append(run_xy_t)

    # 设置画布属性
    #plt.figure(figsize=(cols, rows))
    #plt.figure()
    fig, ax = plt.subplots()

    # 选色
    color_index_ap = {}
    color_chosen_arr_ap = [10, 11, 12, 13, 14, 15]
    i = 0
    for ap_t in ap_list:
        color_index_ap[ap_t] = color_set[color_chosen_arr_ap[i]]
        i += 1

    color_index_traj = []
    interval = 3
    color_chosen_arr_traj = [j for j in range(0, run_xy_yaw_set.__len__() + interval * 2)]
    for i in range(0, run_xy_yaw_set.__len__()):
        color_1 = color_set[color_chosen_arr_traj[i]]
        color_2 = color_set[color_chosen_arr_traj[i + interval]]
        color_3 = color_set[color_chosen_arr_traj[i + interval * 2]]
        color_index_traj.append([color_1, color_2, color_3])

    # 绘制栅格
    grid_edge = grid_size / 40
    for grid_t in grids:
        ap_in_grid_t = grid_t[2]
        if ap_in_grid_t.__len__() == 0:
            #
            x_start = grid_t[1] - grid_size / 2 + grid_edge
            y_start = grid_t[0] - grid_size / 2 + grid_edge
            grid_width = grid_size - grid_edge * 2
            #
            color_t = [0.8, 0.8, 0.8, 0.5]
            #
            ax.add_patch(pc.Rectangle((x_start, y_start), grid_width, grid_width, color=color_t))

    # 绘制AP List
    # 注意覆盖关系
    for grid_t in grids:
        ap_in_grid_t = grid_t[2]
        for i in range(0, ap_list.__len__()):
            for ap_t in ap_in_grid_t:
                if ap_list[i] == ap_t:
                    #
                    x_start = grid_t[1] - grid_size / 2     # x和y是反过来的
                    y_start = grid_t[0] - grid_size / 2
                    #
                    color_t = list(color_index_ap[ap_t]) + [0.5]        # RGB-alpha
                    #
                    ax.add_patch(pc.Rectangle((x_start, y_start), grid_size, grid_size, color=color_t))

    # 轨迹
    traj = []
    x_offset_0 = 0.05
    y_offset_0 = 0.05
    run_number = run_xy_yaw_set.__len__()
    for i in range(0, run_xy_yaw_set.__len__()):
        x_offset =  x_offset_0 * (i - run_number / 2)
        y_offset = -y_offset_0 * (i - run_number / 2)

        x_t       = [x_y_heading_label[0] + x_offset for x_y_heading_label in run_xy_yaw_set[i]]
        y_t       = [x_y_heading_label[1] + y_offset for x_y_heading_label in run_xy_yaw_set[i]]
        heading_t = [x_y_heading_label[2]            for x_y_heading_label in run_xy_yaw_set[i]]

        #
        # method 1
        if not is_gradient_color:
            line_t = ax.plot(y_t, x_t, color=color_index_traj[i][0], lw=3.25, label='run_%d' % (i,))   # x和y是反过来的, 下同
            traj.append(line_t)
        else:
            #
            # method 2
            #
            color_1 = color_index_traj[i][0]
            color_2 = color_index_traj[i][1]
            color_3 = color_index_traj[i][2]
            colors = [color_1, color_2, color_3]            # [color_1, color_2]
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list("", colors)
            points = np.array([y_t, x_t]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, label='run_%d' % (i,))
            lc.set_array(np.linspace(0, 1, len(y_t)))
            lc.set_linewidth(3)
            #
            ax.add_collection(lc)
            traj.append(lc)

        # 标注起点
        # 注意覆盖关系
        circle_r = grid_size / 6 + 0.005 * (run_xy_yaw_set.__len__() - i)
        c = pc.Circle(xy=(y_t[0], x_t[0], ), radius=circle_r, alpha=0.85, color=color_index_traj[i][0])
        ax.add_patch(c)

        #
        # marking for heading
        for j in range(0, run_xy_yaw_set[i].__len__() - 1):
            if x_t[j + 1] == x_t[j] and y_t[j + 1] == y_t[j]:
                ax, polygon = draw_fuselage(ax, (x_t[j], y_t[j]), heading_t[j], face_color=color_index_traj[i][1], alpha=0.5)
            else:
                ax, polygon = draw_fuselage(ax, (x_t[j], y_t[j]), heading_t[j], face_color=color_index_traj[i][1])
        #
        len_t = run_xy_yaw_set[i].__len__() - 1
        ax, polygon = draw_fuselage(ax, (x_t[len_t], y_t[len_t]), heading_t[len_t], face_color=color_index_traj[i][1])

        # 色条
        # cbar = plt.colorbar(lc, ax=ax)
        # cbar.set_label('Time Normalized')

    ax.set_xlim(Ym - grid_size, Ys + grid_size)
    ax.set_ylim(Xm - grid_size, Xs + grid_size)
    ax.set_xticks(np.arange(Ym - grid_size, Ys + grid_size, grid_size))
    ax.set_yticks(np.arange(Xm - grid_size, Xs + grid_size, grid_size))
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_aspect('equal')
    if not is_gradient_color:
        ax.legend()

    return field

def visualiza_in_animation(mdp, s0, XX, LL, UU, MM, name=None, is_illustrate_together=True, is_show_action=False, is_gradient_color=True):
    #
    # XX: 状态序列, list(list()), 一次给了多个run
    # LL: X -> 2^AP 命题序列
    # UU: 控制序列
    # MM: AP是否满足原子命题的序列
    grid_size = 0.5

    def state_2_grid_center(state):
        x_t = state[0]
        y_t = state[1]
        return x_t, y_t

    def xy2sub(len, x, y, grid_size):
        r = int(len - y / grid_size - 1)
        c = int(x / grid_size)
        return [r, c]

    def sub2xy(len, r, c, grid_size):
        x = c * grid_size
        y = len - r * grid_size - 1
        return [x, y]

    # -------------------------------------------
    # 设置画布属性
    # plt.figure(figsize=(cols, rows))
    # plt.figure()
    fig, ax = plt.subplots()

    # step 1 find all grids from motion mdp
    grids = []
    ap_list = []
    for state_t in list(mdp.nodes.keys()):
        #
        # x_t = state_t[0]
        # y_t = state_t[1]
        x_t, y_t = state_2_grid_center(state_t)
        #
        label_t = []
        act_t = []
        for key_t in mdp.nodes[state_t]:
            if key_t == 'label':
                label_t = list(mdp.nodes[state_t]['label'].keys())[0]  # 这个是试出来的
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
    ap_list.sort()

    Ys = max(x_y_ap[1] for x_y_ap in grids)
    Xs = max(x_y_ap[0] for x_y_ap in grids)
    Ym = min(x_y_ap[1] for x_y_ap in grids)
    Xm = min(x_y_ap[0] for x_y_ap in grids)

    # 其中X和矩阵地图的cols对应
    rows = int(Ys / grid_size)
    cols = int(Xs / grid_size)
    # 创建全部为空地的地图栅格，其中空地以数字1表征
    # ！！！注意ones(行列个数，因此rows需要+1)
    field = np.ones([rows, cols])

    run_xy_yaw_set = []
    for run_t in XX:
        run_xy_t = []
        for state_t in run_t:
            x_t, y_t = state_2_grid_center(state_t)
            heading_t = state_t[2]
            run_xy_t.append((x_t, y_t, heading_t,))
        run_xy_yaw_set.append(run_xy_t)

    if is_show_action:
        is_illustrate_together = False
        next_action_set = []                # [{state_1 : [action, probability, cost], state_2 : [action, probability, cost], ...}, {state_3 : [], state_4 :  [], state_5 :  [], ...}, ...]
        #
        # to obtain action set
        for i in range(0, XX.__len__()):
            run_t = XX[i]
            next_action_set_t = []
            for j in range(0, run_t.__len__() - 1):
                current_state_t  = run_t[j]
                current_action_t = UU[i][j]
                consequence_set_t = mdp[current_state_t]
                #
                next_action_t = {}
                for c_t in consequence_set_t:
                    for action_t in consequence_set_t[c_t]['prop'].keys():
                        probability_t = consequence_set_t[c_t]['prop'][action_t][0]
                        cost_t = consequence_set_t[c_t]['prop'][action_t][1]
                        next_state_t = list
                        if action_t == current_action_t:
                            # to find all possible consequences under current actions
                            next_action_t[c_t] = [action_t, probability_t, cost_t]
                #
                next_action_set_t.append(next_action_t)
            next_action_set.append(next_action_set_t)


    # 选色
    color_index_ap = {}
    color_chosen_arr_ap = [10, 11, 12, 13, 14, 15]
    i = 0
    for ap_t in ap_list:
        color_index_ap[ap_t] = color_set[color_chosen_arr_ap[i]]
        i += 1

    color_index_traj = []
    interval = 3
    color_chosen_arr_traj = [j for j in range(0, run_xy_yaw_set.__len__() + interval * 2)]
    for i in range(0, run_xy_yaw_set.__len__()):
        color_1 = color_set[color_chosen_arr_traj[i]]
        color_2 = color_set[color_chosen_arr_traj[i + interval]]
        color_3 = color_set[color_chosen_arr_traj[i + interval * 2]]
        color_index_traj.append([color_1, color_2, color_3])

        # 绘制栅格
        grid_edge = grid_size / 40
        for grid_t in grids:
            ap_in_grid_t = grid_t[2]
            if ap_in_grid_t.__len__() == 0:
                #
                x_start = grid_t[1] - grid_size / 2 + grid_edge
                y_start = grid_t[0] - grid_size / 2 + grid_edge
                grid_width = grid_size - grid_edge * 2
                #
                color_t = [0.8, 0.8, 0.8, 0.5]
                #
                ax.add_patch(pc.Rectangle((x_start, y_start), grid_width, grid_width, color=color_t))

        # 绘制AP List
        # 注意覆盖关系
        for grid_t in grids:
            ap_in_grid_t = grid_t[2]
            for i in range(0, ap_list.__len__()):
                for ap_t in ap_in_grid_t:
                    if ap_list[i] == ap_t:
                        #
                        x_start = grid_t[1] - grid_size / 2  # x和y是反过来的
                        y_start = grid_t[0] - grid_size / 2
                        #
                        color_t = list(color_index_ap[ap_t]) + [0.5]  # RGB-alpha
                        #
                        ax.add_patch(pc.Rectangle((x_start, y_start), grid_size, grid_size, color=color_t))


    # ----------------------------------------------------
    #
    #
    frame_num = min([run_t.__len__() for run_t in run_xy_yaw_set])
    fps = 25                                                 # alternative, 20, debugging = 5
    frame_num = frame_num * fps
    if not is_illustrate_together:
        frame_num = frame_num * run_xy_yaw_set.__len__()    # multiplies the number of runs

    # 轨迹
    traj = []                           # in fact, it is feasible to set traj local, not global
    action_in_traj = []

    # 清空当前帧
    def init():
        nonlocal ax, traj
        nonlocal run_xy_yaw_set, next_action_set
        traj.clear()
        return traj

    # 更新新一帧的数据
    def update(frame):
        nonlocal ax, is_illustrate_together, is_show_action
        nonlocal run_xy_yaw_set, next_action_set
        x_offset_0 = 0.05
        y_offset_0 = 0.05

        run_number = run_xy_yaw_set.__len__()
        run_sequence_number = min([run_xy_yaw_set_t.__len__() for run_xy_yaw_set_t in run_xy_yaw_set])

        if is_illustrate_together:

            for i in range(0, run_xy_yaw_set.__len__()):
                x_offset = x_offset_0 * (i - run_number / 2)
                y_offset = -y_offset_0 * (i - run_number / 2)

                # x_t = [x_y_label[0] + x_offset for x_y_label in run_xy_yaw_set[i]]
                # y_t = [x_y_label[1] + y_offset for x_y_label in run_xy_yaw_set[i]]
                x_t = []
                y_t = []
                heading_t = []
                current_index_2_display = int(math.floor(frame / fps))
                if current_index_2_display < 1:
                    current_index_2_display = 1
                for j in range(0, current_index_2_display):
                    x_y_label = run_xy_yaw_set[i][j]
                    x_t.append(x_y_label[0])
                    y_t.append(x_y_label[1])
                    heading_t.append(x_y_label[2])

                if not is_gradient_color:
                    #
                    # method 1
                    line_t = ax.plot(y_t, x_t, color=color_index_traj[i][0], lw=3, label='run_%d' % (i,)) # x和y是反过来的, 下同
                    traj.append(line_t[0])
                else:
                    #
                    # method 2
                    #
                    color_1 = color_index_traj[i][0]
                    color_2 = color_index_traj[i][1]
                    color_3 = color_index_traj[i][2]
                    colors = [color_1, color_2, color_3]  # [color_1, color_2]
                    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("", colors)
                    points = np.array([y_t, x_t]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap=cmap, label='run_%d' % (i,))
                    lc.set_array(np.linspace(0, 1, len(y_t)))
                    lc.set_linewidth(3)
                    #
                    ax.add_collection(lc)
                    traj.append(lc)

                # 标注起点
                # 注意覆盖关系
                circle_r = grid_size / 6 + 0.005 * (run_xy_yaw_set.__len__() - i)
                c = pc.Circle(xy=(y_t[0], x_t[0],), radius=circle_r, alpha=0.85, color=color_index_traj[i][0])
                ax.add_patch(c)
                traj.append(c)

                #
                # marking for possible actions

                #
                # marking for heading
                for j in range(0, current_index_2_display - 1):
                    if j + 1 <= current_index_2_display - 1 and x_t[j + 1] == x_t[j] and y_t[j + 1] == y_t[j]:
                        ax, polygon = draw_fuselage(ax, (x_t[j], y_t[j]), heading_t[j], face_color=color_index_traj[i][1],
                                                    alpha=0.5)
                    else:
                        ax, polygon = draw_fuselage(ax, (x_t[j], y_t[j]), heading_t[j], face_color=color_index_traj[i][1])
                #
                ax, polygon = draw_fuselage(ax, (x_t[current_index_2_display - 1], y_t[current_index_2_display - 1]), heading_t[current_index_2_display - 1], face_color=color_index_traj[i][1])
                traj.append(polygon)

                # 色条
                # cbar = plt.colorbar(lc, ax=ax)
                # cbar.set_label('Time Normalized')

            # to delete historical fuselages
            count_valid_ploy_num = 0
            for i in range(len(traj) - 1, -1, -1):
                if type(traj[i]) == matplotlib.patches.Polygon:
                    count_valid_ploy_num += 1
                    if count_valid_ploy_num > run_number:
                        del traj[i]


        else:
            current_index_2_display = int(math.floor(frame / fps))
            current_agent   = int(current_index_2_display / run_sequence_number)            # index for current agent
            current_index_i = int(current_index_2_display) % run_sequence_number

            traj.clear()

            x_t = []
            y_t = []
            heading_t = []
            if current_index_i < 1:
                current_index_i = 1
            for j in range(0, current_index_i):
                x_y_label = run_xy_yaw_set[current_agent][j]
                x_t.append(x_y_label[0])
                y_t.append(x_y_label[1])
                heading_t.append(x_y_label[2])

            if not is_gradient_color:
                #
                # method 1
                line_t = ax.plot(y_t, x_t, color=color_index_traj[current_agent][0], lw=3, label='run_%d' % (current_agent,))  # x和y是反过来的, 下同
                traj.append(line_t[0])
            else:
                #
                # method 2
                #
                color_1 = color_index_traj[current_agent][0]
                color_2 = color_index_traj[current_agent][1]
                color_3 = color_index_traj[current_agent][2]
                colors = [color_1, color_2, color_3]  # [color_1, color_2]
                cmap = plt.cm.colors.LinearSegmentedColormap.from_list("", colors)
                points = np.array([y_t, x_t]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=cmap, label='run_%d' % (current_agent,))
                lc.set_array(np.linspace(0, 1, len(y_t)))
                lc.set_linewidth(3)
                #
                ax.add_collection(lc)
                traj.append(lc)

            # 标注起点
            # 注意覆盖关系
            circle_r = grid_size / 6
            c = pc.Circle(xy=(y_t[0], x_t[0],), radius=circle_r, alpha=0.85, color=color_index_traj[current_agent][0])
            ax.add_patch(c)
            traj.append(c)

            # to delete historical fuselages
            count_valid_ploy_num = 0
            for i in range(len(traj) - 1, -1, -1):
                if type(traj[i]) == matplotlib.patches.Polygon:
                    count_valid_ploy_num += 1
                    if count_valid_ploy_num > run_number:
                        del traj[i]

            ax, polygon = draw_fuselage(ax, (x_t[current_index_i - 1], y_t[current_index_i - 1]), heading_t[current_index_i - 1], face_color=color_index_traj[current_agent][1])
            traj.append(polygon)

            if is_show_action:
                if current_index_i <= run_sequence_number - 1 - 1:
                    text_offset_x = 0.125
                    k = 0
                    for state_t in next_action_set[current_agent][current_index_i - 1].keys():
                        x_next_t = state_t[0]
                        y_next_t = state_t[1]
                        heading_next_t = state_t[2]
                        #
                        action_t      = next_action_set[current_agent][current_index_i - 1][state_t][0]
                        probability_t = next_action_set[current_agent][current_index_i - 1][state_t][1]
                        cost_t        = next_action_set[current_agent][current_index_i - 1][state_t][2]

                        x_t_t = x_t[current_index_i - 1]
                        y_t_t = y_t[current_index_i - 1]

                        x_text_t = x_next_t
                        y_text_t = y_next_t
                        if x_text_t > x_t_t:
                            x_text_t += 0.25
                        elif x_text_t < x_t_t:
                            x_text_t -= 0.25
                        x_text_t = x_text_t + text_offset_x * (k - 1)

                        if y_text_t > y_t_t:
                            y_text_t += 0.25
                        elif y_text_t < y_t_t:
                            y_text_t -= 0.25


                        #
                        line_t = ax.plot([y_t_t, y_next_t], [x_t_t, x_next_t], color='gray', lw=2.5, linestyle='--', alpha=0.375)
                        ax, polygon = draw_fuselage(ax, (x_next_t, y_next_t), heading_next_t, edge_color=color_index_traj[current_agent][2], face_color=color_index_traj[current_agent][2], alpha=0.45)
                        text_t = ax.text(y_text_t, x_text_t, "(%s, %f, %f)" % (str(list(action_t)), probability_t, cost_t,), fontsize=8.5)

                        #
                        traj.append(line_t[0])
                        traj.append(polygon)
                        traj.append(text_t)

                        k += 1

            # 色条
            # cbar = plt.colorbar(lc, ax=ax)
            # cbar.set_label('Time Normalized')

        return traj

    ax.set_xlim(Ym - grid_size, Ys + grid_size)
    ax.set_ylim(Xm - grid_size, Xs + grid_size)
    ax.set_xticks(np.arange(Ym - grid_size, Ys + grid_size, grid_size))
    ax.set_yticks(np.arange(Xm - grid_size, Xs + grid_size, grid_size))
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_aspect('equal')
    if not is_gradient_color:
        ax.legend()

    # 调用 FuncAnimation
    ani = FuncAnimation(fig
                        , update
                        , init_func=init
                        , frames=frame_num
                        , interval=2
                        , blit=True
                        )

    return ani

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
