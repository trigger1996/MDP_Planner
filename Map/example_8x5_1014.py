from MDP_TG.mdp import Motion_MDP
import pickle
import time

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib import colors
import matplotlib.patches as pc
import matplotlib.ticker as ticker
from matplotlib.style import available


grid_size = 0.5
ws_x_max = 8
ws_y_max = 5

observed_area = {
    'A'   : [ (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4) ],
    'B'   : [ (0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (4, 2) ],
    'C'   : [ (4, 3), (5, 3), (5, 4) ],
    'D'   : [ (4, 1), (5, 1), (5, 2), (6, 1), (6, 2) ],
    'E'   : [ (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (7, 1), (7, 2), (6, 3), (6, 4), (7, 3), (7, 4) ],
    'OBS' : [ (3, 1), (3, 2), (3, 3) ]
}

# to preset color and fonts
config = {
    "font.family":'serif',
    "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'], # simsun字体中文版就是宋体
    'text.usetex': True,
}
rcParams.update(config)

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

def xy_2_grid(x, y):
    global grid_size
    x_grid = (x - grid_size / 2) / grid_size
    y_grid = (y - grid_size / 2) / grid_size
    return int(x_grid), int(y_grid)

def grid_2_xy(x_grid, y_grid):
    global grid_size
    x = x_grid * grid_size + grid_size / 2
    y = y_grid * grid_size + grid_size / 2
    return x, y

#
# https://blog.csdn.net/qq_42727752/article/details/117339892
def state_2_grid_center(state):
    x_t = state[0]
    y_t = state[1]
    return x_t, y_t

def observation_func(state):
    global observed_area
    x = state[0][0]
    y = state[0][1]

    x_grid, y_grid = xy_2_grid(x, y)
    for observed_state_t in observed_area.keys():
        if (x_grid, y_grid) in observed_area[observed_state_t]:
            return observed_state_t

    return None

def build_model(is_enable_single_direction=True):
    global grid_size
    global ws_x_max, ws_y_max

    t0 = time.time()

    # (0, 0) is the origin of the map
    grids = {
        #
        (2, 4): {frozenset(['supply']):   0.6,  frozenset(): 0.0, },
        #
        (6, 3): {frozenset(['supply']):   0.9,  frozenset(): 0.0, },
        (6, 4): {frozenset(['supply']):   0.9,  frozenset(): 0.0, },
        (7, 3): {frozenset(['supply']):   0.9,  frozenset(): 0.0, },
        (7, 4): {frozenset(['supply']):   0.9,  frozenset(): 0.0, },
        #
        (0, 0): {frozenset(['recharge']): 0.85, frozenset(): 0.0, },
        (7, 0): {frozenset(['recharge']): 0.75, frozenset(): 0.0, },
        #
        (2, 2): {frozenset(['drop']):     0.75, frozenset(): 0.0, },
        (4, 2): {frozenset(['drop']):     0.75, frozenset(): 0.0, },
        #
        (3, 1): {frozenset(['obstacle']): 1.0,  frozenset(): 0.0, },
        (3, 2): {frozenset(['obstacle']): 1.0,  frozenset(): 0.0, },
        (3, 3): {frozenset(['obstacle']): 1.0,  frozenset(): 0.0, },
    }

    # append states with no aps to ws dicts
    for x_grid in range(0, ws_x_max):
        for y_grid in range(0, ws_y_max):
            xy_grids = (x_grid, y_grid)
            if xy_grids not in grids:
                grids[xy_grids] = {frozenset(): 1.0, }


    states = dict()
    for grid_t in grids.keys():
        x_grid = grid_t[0]
        y_grid = grid_t[1]
        x, y = grid_2_xy(x_grid, y_grid)
        states[(x, y)] = grids[grid_t]


    print('WS_node_dict_size', len(states))

    pickle.dump((states, grid_size), open('ws_model_8x5_1014.p', "wb"))
    print('ws_model.p saved!')
    # ----
    # visualize_world(WS_d, WS_node_dict, 'world')
    # t1 = time.time()
    # print 'visualize world done, time: %s' %str(t1-t0)
    # ------------------------------------
    robot_nodes = dict()
    for loc, prop in states.items():
        for d in ['N', 'S', 'E', 'W']:
            robot_nodes[(loc[0], loc[1], d)] = prop
    # ------------------------------------
    initial_grid =  (1, 1)
    initial_heading = 'N'
    x_initial, y_initial = grid_2_xy(initial_grid[0], initial_grid[1])
    initial_node = (x_initial, y_initial, initial_heading)
    initial_label = frozenset()

    U = [tuple('FR'), tuple('BK'), tuple('TR'), tuple('TL'), tuple('ST')]
    C = [2, 4, 3, 3, 1]
    P_FR = [0.1, 0.8, 0.1]
    P_BK = [0.15, 0.7, 0.15]
    P_TR = [0.05, 0.9, 0.05]
    P_TL = [0.05, 0.9, 0.05]
    P_ST = [0.005, 0.99, 0.005]
    # -------------
    robot_edges = dict()
    for fnode in robot_nodes.keys():
        fx = fnode[0]
        fy = fnode[1]
        fd = fnode[2]
        # action FR
        u = U[0]
        c = C[0]
        if fd == 'N':
            t_nodes = [(fx-grid_size, fy+grid_size, fd), (fx, fy+grid_size, fd),
                       (fx+grid_size, fy+grid_size, fd)]
        if fd == 'S':
            t_nodes = [(fx-grid_size, fy-grid_size, fd), (fx, fy-grid_size, fd),
                       (fx+grid_size, fy-grid_size, fd)]
        if fd == 'E':
            t_nodes = [(fx+grid_size, fy-grid_size, fd), (fx+grid_size, fy, fd),
                       (fx+grid_size, fy+grid_size, fd)]
        if fd == 'W':
            t_nodes = [(fx-grid_size, fy-grid_size, fd), (fx-grid_size, fy, fd),
                       (fx-grid_size, fy+grid_size, fd)]
        for k, tnode in enumerate(t_nodes):
            if tnode in list(robot_nodes.keys()):
                robot_edges[(fnode, u, tnode)] = (P_FR[k], c)
        # action BK
        u = U[1]
        c = C[1]
        if fd == 'N':
            t_nodes = [(fx-grid_size, fy-grid_size, fd), (fx, fy-grid_size, fd),
                       (fx+grid_size, fy-grid_size, fd)]
        if fd == 'S':
            t_nodes = [(fx-grid_size, fy+grid_size, fd), (fx, fy+grid_size, fd),
                       (fx+grid_size, fy+grid_size, fd)]
        if fd == 'E':
            t_nodes = [(fx-grid_size, fy-grid_size, fd), (fx-grid_size, fy, fd),
                       (fx-grid_size, fy+grid_size, fd)]
        if fd == 'W':
            t_nodes = [(fx+grid_size, fy-grid_size, fd), (fx+grid_size, fy, fd),
                       (fx+grid_size, fy+grid_size, fd)]
        for k, tnode in enumerate(t_nodes):
            if tnode in list(robot_nodes.keys()):
                robot_edges[(fnode, u, tnode)] = (P_BK[k], c)
        # action TR
        u = U[2]
        c = C[2]
        if fd == 'N':
            t_nodes = [(fx, fy, 'N'), (fx, fy, 'E'), (fx, fy, 'S')]
        if fd == 'S':
            t_nodes = [(fx, fy, 'S'), (fx, fy, 'W'), (fx, fy, 'N')]
        if fd == 'E':
            t_nodes = [(fx, fy, 'E'), (fx, fy, 'S'), (fx, fy, 'W')]
        if fd == 'W':
            t_nodes = [(fx, fy, 'W'), (fx, fy, 'N'), (fx, fy, 'E')]
        for k, tnode in enumerate(t_nodes):
            if tnode in list(robot_nodes.keys()):
                robot_edges[(fnode, u, tnode)] = (P_TR[k], c)
        # action TL
        u = U[3]
        c = C[3]
        if fd == 'S':
            t_nodes = [(fx, fy, 'S'), (fx, fy, 'E'), (fx, fy, 'N')]
        if fd == 'N':
            t_nodes = [(fx, fy, 'N'), (fx, fy, 'W'), (fx, fy, 'S')]
        if fd == 'W':
            t_nodes = [(fx, fy, 'W'), (fx, fy, 'S'), (fx, fy, 'E')]
        if fd == 'E':
            t_nodes = [(fx, fy, 'E'), (fx, fy, 'N'), (fx, fy, 'W')]
        for k, tnode in enumerate(t_nodes):
            if tnode in list(robot_nodes.keys()):
                robot_edges[(fnode, u, tnode)] = (P_TL[k], c)
        # action ST
        '''
        u = U[4]
        c = C[4]
        if fd == 'S':
            t_nodes = [(fx, fy, 'W'), (fx, fy, 'S'), (fx, fy, 'E')]
        if fd == 'N':
            t_nodes = [(fx, fy, 'W'), (fx, fy, 'N'), (fx, fy, 'E')]
        if fd == 'W':
            t_nodes = [(fx, fy, 'S'), (fx, fy, 'W'), (fx, fy, 'N')]
        if fd == 'E':
            t_nodes = [(fx, fy, 'N'), (fx, fy, 'E'), (fx, fy, 'S')]
        for k, tnode in enumerate(t_nodes):
            if tnode in list(robot_nodes.keys()):
                robot_edges[(fnode, u, tnode)] = (P_ST[k], c)
        '''

    # TODO
    # remove edges from obstacles
    # the transitions to obstacles are allowed for testing whether our algorithm can guarantee safety of the system
    pass
    # remove inaccessible nodes
    inaccessible_edges_in_grids = [
        # single directions
        [ (3, 4), (2, 4) ],
        [ (4, 4), (3, 4) ],
        [ (3, 3), (3, 4) ],
        [ (4, 3), (3, 4) ],
        [ (3, 4), (2, 3) ],
        #
        [ (3, 3), (2, 4) ],
        [ (4, 4), (3, 3) ],
        # single direction 2
        [ (2, 0), (3, 0) ],
        [ (3, 0), (4, 0) ],
        [ (3, 1), (3, 0) ],
        [ (2, 1), (3, 0) ],
        [ (3, 0), (4, 1) ],
        #
        [ (3, 1), (4, 0) ],
        [ (2, 0), (3, 1) ],
    ]
    if is_enable_single_direction:
        edges_to_remove = []
        for edge_in_xy_t in robot_edges.keys():
            #
            x = edge_in_xy_t[0][0]
            y = edge_in_xy_t[0][1]
            #
            x_p = edge_in_xy_t[2][0]
            y_p = edge_in_xy_t[2][1]
            #
            x_grid,   y_grid   = xy_2_grid(x, y)
            x_grid_p, y_grid_p = xy_2_grid(x_p, y_p)

            if [ (x_grid, y_grid), (x_grid_p, y_grid_p) ] in inaccessible_edges_in_grids:
                edges_to_remove.append(edge_in_xy_t)

        for edge_t in edges_to_remove:
            del robot_edges[edge_t]

    # ----
    ws_robot_model = (robot_nodes, robot_edges, U,
                      initial_node, initial_label)
    t1 = time.time()
    print('ws_robot_model returned, time: %s' % str(t1-t0))
    return ws_robot_model

#
# Visualization ------------------------------------------------------------------------------------------------------
def draw_fuselage(ax, center, heading, face_color='lightblue', edge_color='darkblue', alpha=1.):
    yaw = 0
    if type(heading) == int or type(heading) == float:
        yaw = heading
    elif type(heading) == str:
        if heading == 'N':
            yaw = -math.pi / 2
        elif heading == 'E':
            yaw =  0
        elif heading == 'S':
            yaw = math.pi / 2
        elif heading == 'W':
            yaw =  math.pi

    # 五边形的顶点坐标

    vertices = np.array([
        (0.03,   0.0),
        (0,      0.05),
        (-0.05,  0.05),
        (-0.05, -0.05),
        (0,     -0.05)
    ])                          # (y, x)

    for i in range(0, vertices.shape[0]):
        x_t = vertices[i,0]
        y_t = vertices[i,1]
        vertices[i, 0] = x_t *  math.cos(yaw) + y_t * math.sin(yaw) + center[0]         # x
        vertices[i, 1] = x_t * -math.sin(yaw) + y_t * math.cos(yaw) + center[1]         # y

    # 创建五边形
    polygon = Polygon(vertices, closed=True, edgecolor=edge_color, facecolor=face_color, alpha=alpha)
    ax.add_patch(polygon)

    return ax, polygon


def visualiza_in_animation(mdp, s0, XX, LL, UU, MM, name=None, is_illustrate_together=True, is_show_action=False, is_gradient_color=True):
    global grid_size
    #
    # XX: 状态序列, list(list()), 一次给了多个run
    # LL: X -> 2^AP 命题序列
    # UU: 控制序列
    # MM: AP是否满足原子命题的序列



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

    Xs = max(x_y_ap[0] for x_y_ap in grids)
    Ys = max(x_y_ap[1] for x_y_ap in grids)
    Xm = min(x_y_ap[0] for x_y_ap in grids)
    Ym = min(x_y_ap[1] for x_y_ap in grids)

    # 其中X和矩阵地图的cols对应
    rows = int(Xs / grid_size)
    cols = int(Ys / grid_size)
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
                x_start = grid_t[0] - grid_size / 2 + grid_edge
                y_start = grid_t[1] - grid_size / 2 + grid_edge
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
                        x_start = grid_t[0] - grid_size / 2  # x和y是反过来的
                        y_start = grid_t[1] - grid_size / 2
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
                x_offset =  x_offset_0 * (i - run_number / 2)
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
                    points = np.array([x_t, y_t]).T.reshape(-1, 1, 2)
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
                c = pc.Circle(xy=(x_t[0], y_t[0],), radius=circle_r, alpha=0.85, color=color_index_traj[i][0])
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
                points = np.array([x_t, y_t]).T.reshape(-1, 1, 2)
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
            c = pc.Circle(xy=(x_t[0], y_t[0],), radius=circle_r, alpha=0.85, color=color_index_traj[current_agent][0])
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
                    text_offset_y = 0.125
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

                        if y_text_t > y_t_t:
                            y_text_t += 0.25
                        elif y_text_t < y_t_t:
                            y_text_t -= 0.25
                        y_text_t = y_text_t + text_offset_y * (k - 1)

                        #
                        line_t = ax.plot([x_t_t, x_next_t], [y_t_t, y_next_t], color='gray', lw=2.5, linestyle='--', alpha=0.375)
                        ax, polygon = draw_fuselage(ax, (x_next_t, y_next_t), heading_next_t, edge_color=color_index_traj[current_agent][2], face_color=color_index_traj[current_agent][2], alpha=0.45)
                        text_t = ax.text(x_text_t, y_text_t, "(%s, %f, %f)" % (str(list(action_t)), probability_t, cost_t,), fontsize=8.5)

                        #
                        traj.append(line_t[0])
                        traj.append(polygon)
                        traj.append(text_t)

                        k += 1

            # 色条
            # cbar = plt.colorbar(lc, ax=ax)
            # cbar.set_label('Time Normalized')

        return traj

    ax.set_xlim(Xm - grid_size / 2, Xs + grid_size / 2)
    ax.set_ylim(Ym - grid_size / 2, Ys + grid_size / 2)
    ax.set_xticks(np.arange(Xm - grid_size / 2, Xs + grid_size / 2, grid_size))
    ax.set_yticks(np.arange(Ym - grid_size / 2, Ys + grid_size / 2, grid_size))
    ax.set_xlabel(r'$x(m)$')
    ax.set_ylabel(r'$y(m)$')
    ax.set_aspect('equal')
    if not is_gradient_color:
        ax.legend()

    # 调用 FuncAnimation
    ani = FuncAnimation(fig
                        , update
                        , init_func=init
                        , frames=frame_num
                        , interval=10
                        , blit=True
                        )

    return ani


def visualize_trajectories(mdp, s0, XX, LL, UU, MM, name=None, is_gradient_color=True):
    global grid_size
    #
    # XX: 状态序列, list(list()), 一次给了多个run
    # LL: X -> 2^AP 命题序列
    # UU: 控制序列
    # MM: AP是否满足原子命题的序列

    # step 1 find all grids from motion mdp
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

    Xs = max(x_y_ap[0] for x_y_ap in grids)
    Ys = max(x_y_ap[1] for x_y_ap in grids)
    Xm = min(x_y_ap[0] for x_y_ap in grids)
    Ym = min(x_y_ap[1] for x_y_ap in grids)

    # 其中X和矩阵地图的cols对应
    rows = int(Xs / grid_size)
    cols = int(Ys / grid_size)
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
            x_start = grid_t[0] - grid_size / 2 + grid_edge
            y_start = grid_t[1] - grid_size / 2 + grid_edge
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
                    x_start = grid_t[0] - grid_size / 2     # x和y是反过来的
                    y_start = grid_t[1] - grid_size / 2
                    #
                    color_t = list(color_index_ap[ap_t]) + [0.5]        # RGB-alpha
                    #
                    ax.add_patch(pc.Rectangle((x_start, y_start), grid_size, grid_size, color=color_t))

    # 轨迹
    traj = []
    x_offset_0 = 0.0
    y_offset_0 = 0.0
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
            line_t = ax.plot(x_t, y_t, color=color_index_traj[i][0], lw=3.25, label='run_%d' % (i,))   # x和y是反过来的, 下同
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
            points = np.array([x_t, y_t]).T.reshape(-1, 1, 2)
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
        c = pc.Circle(xy=(x_t[0], y_t[0], ), radius=circle_r, alpha=0.85, color=color_index_traj[i][0])
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

    ax.set_xlim(Xm - grid_size / 2, Xs + grid_size / 2)
    ax.set_ylim(Ym - grid_size / 2, Ys + grid_size / 2)
    ax.set_xticks(np.arange(Xm - grid_size / 2, Xs + grid_size / 2, grid_size))
    ax.set_yticks(np.arange(Ym - grid_size / 2, Ys + grid_size / 2, grid_size))
    ax.set_xlabel(r'$x(m)$')
    ax.set_ylabel(r'$y(m)$')
    ax.set_aspect('equal')
    if not is_gradient_color:
        ax.legend()

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

def draw_mdp_principle():
    global grid_size
    fig, ax = plt.subplots(figsize=(10.5, 7))

    # draw grids
    #
    grid_xy_area = [[-1, 3], [-1, 1]]       # [[x_min, x_max], [y_min, y_max]]
    grid_edge = grid_size / 40
    for x_grid in range(grid_xy_area[0][0], grid_xy_area[0][1] + 1):
        for y_grid in range(grid_xy_area[1][0], grid_xy_area[1][1] + 1):
            x, y = grid_2_xy(x_grid, y_grid)
            x_start = x - grid_size / 2 + grid_edge
            y_start = y - grid_size / 2 + grid_edge
            grid_width = grid_size - grid_edge * 2
            #
            color_t = [0.8, 0.8, 0.8, 0.5]
            #
            ax.add_patch(pc.Rectangle((x_start, y_start), grid_width, grid_width, color=color_t))


    state_seq = [(0, 0, 'N'),
                 (0, 1, 'N'),
                 (0, 1, 'E'),
                 (1, 1, 'E'),
                 (2, 0, 'E'),
                 ]
    possible_action_seq = [['FR', (-1, 1,       'N'), (0,       1,       'N'), (1, 1, 'N'), ],
                           ['RT', (0,  1 + 0.1, 'N'), (0 + 0.1, 1 + 0.1, 'E'), (0,  1 + 0.1,   'S'),],
                           ['FR', (1, 1,        'E'), (1,       0,       'E'),],
                           ['FR', (2, 1,        'E'), (2,       0,       'E'),],
                           ['FR', (3, 1,        'E'), (3,       0,       'E'), (3, -1, 'E'), ], ]

    color_0 = color_set[0]
    color_motion  = color_set[22]
    color_actions = [[color_set[5],  color_set[20],  color_set[15]],
                     [color_set[6],  color_set[11],  color_set[16]],
                     [color_set[7],  color_set[12],  color_set[17]],
                     [color_set[8],  color_set[13],  color_set[18]],
                     [color_set[9],  color_set[14],  color_set[21]],
                     ]

    unit_t = grid_size
    #                    act,   x,       y,       pr,   x,      y,       pr,  x,      y          pr,   x,     y
    text_xy_actions = [[('FR',  0,                    -unit_t,), (0.1,  -2.5  * unit_t, 1.5  * unit_t,),   (0.8,  1.5  * unit_t, -0.95 * unit_t,), (0.1,  0.95  * unit_t, 1.55 * unit_t,),],
                       [('TR', -1.55  * unit_t,        unit_t,), (0.15, -1.25 * unit_t, 2.25 * unit_t,),   (0.7, -1.25 * unit_t,  2.05 * unit_t,), (0.15, 2.25 * unit_t, 0.5  * unit_t,),],
                       [('FR',  2.25  * unit_t, 2.0 *  unit_t,), (0.9,   2.05 * unit_t, 1.25  * unit_t,),  (0.1,  2.05 * unit_t, -0.95 * unit_t,), ],
                       [('FR',  4.25  * unit_t, 0.5 *  unit_t,), (0.9,   4.05 * unit_t, 1.25  * unit_t,),  (0.1,  4.05 * unit_t, -0.95 * unit_t,), ],
                       [('FR',  6.25  * unit_t, 0.5 *  unit_t,), (0.1,   6.05 * unit_t, 1.25  * unit_t,),  (0.9,  6.05 * unit_t, -0.95 * unit_t,), (0.1,  6.05 * unit_t, -2.25 * unit_t,), ],]

    for i in range(0, state_seq.__len__()):
        x_t, y_t = grid_2_xy(state_seq[i][0], state_seq[i][1])
        heading_t = state_seq[i][2]

        # action
        if i > 0:
            x_t_last, y_t_last = grid_2_xy(state_seq[i - 1][0], state_seq[i - 1][1])

            ax.plot([x_t_last, x_t], [y_t_last, y_t], color=color_motion, lw=5.5, linestyle='-')

        # position
        # 注意遮盖
        draw_fuselage(ax, (x_t, y_t), heading_t, face_color=color_0)

        #
        action_t = text_xy_actions[i][0][0]
        x_text_t, y_text_t = grid_2_xy(text_xy_actions[i][0][1], text_xy_actions[i][0][2])
        ax.text(x_text_t, y_text_t, r"\textbf{%s}" % (action_t, ), color=color_motion, fontsize=25)

        for j in range(1, possible_action_seq[i].__len__()):
            x_next_t, y_next_t = grid_2_xy(possible_action_seq[i][j][0], possible_action_seq[i][j][1])
            heading_next_t = possible_action_seq[i][j][2]

            draw_fuselage(ax, (x_next_t, y_next_t), heading_next_t, face_color=color_actions[i][j - 1], alpha=0.375)

            ax.plot([x_t, x_next_t], [y_t, y_next_t], color=color_actions[i][j - 1], lw=3.5, linestyle='--', alpha=0.375)

            #
            action_t = text_xy_actions[i][j][0]
            x_text_t, y_text_t = grid_2_xy(text_xy_actions[i][j][1], text_xy_actions[i][j][2])
            ax.text(x_text_t, y_text_t, r"\textbf{%s}" % (action_t, ), color=color_actions[i][j - 1], fontsize=17.5)

    #
    x_min, y_min = grid_2_xy(grid_xy_area[0][0], grid_xy_area[1][0])
    x_max, y_max = grid_2_xy(grid_xy_area[0][1], grid_xy_area[1][1])
    ax.set_xlim(x_min - grid_size / 2, x_max + grid_size / 2)
    ax.set_ylim(y_min - grid_size / 2, y_max + grid_size / 2)
    #
    ax.xaxis.set_major_locator(ticker.MultipleLocator(grid_size))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(grid_size))
    #
    ax.set_xlabel(r'$x(m)$')
    ax.set_ylabel(r'$y(m)$')
    ax.set_aspect('equal')

def draw_action_principle():
    '''
    # method 1
    fig, axs = plt.subplots(1, 4)  # 创建2行2列的子图网格

    # 在第1个子图上绘图
    axs[0, 0].plot([1, 2, 3], [4, 5, 6])
    axs[0, 0].set_title('Plot 1')

    # 在第2个子图上绘图
    axs[0, 1].plot([1, 2, 3], [6, 5, 4])
    axs[0, 1].set_title('Plot 2')

    # 在第3个子图上绘图
    axs[1, 0].plot([1, 2, 3], [1, 2, 1])
    axs[1, 0].set_title('Plot 3')

    # 在第4个子图上绘图
    axs[1, 1].plot([1, 2, 3], [3, 3, 3])
    axs[1, 1].set_title('Plot 4')

    # method 2
    fig, axs = plt.subplots(2, 2)  # 创建2行2列的子图网格

    # 在第1个子图上绘图
    #axs[0, 0].plot([1, 2, 3], [4, 5, 6])
    axs[0, 0].set_title('Plot 1')

    # 在第2个子图上绘图
    #axs[0, 1].plot([1, 2, 3], [6, 5, 4])
    axs[0, 1].set_title('Plot 2')

    # 在第3个子图上绘图
    #axs[1, 0].plot([1, 2, 3], [1, 2, 1])
    axs[1, 0].set_title('Plot 3')

    # 在第4个子图上绘图
    #axs[1, 1].plot([1, 2, 3], [3, 3, 3])
    axs[1, 1].set_title('Plot 4')
    '''

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # 创建2行2列的子图网格
    axs_fr = axs[0, 0]
    axs_bk = axs[1, 0]
    axs_lt = axs[0, 1]
    axs_rt = axs[1, 1]

    grid_size = 0.25
    #
    grids = [[0,   0],         [grid_size,  0],         [-grid_size,  0],
             [0,   grid_size], [grid_size,  grid_size], [-grid_size,  grid_size],
             [0,  -grid_size], [grid_size, -grid_size], [-grid_size, -grid_size]]
    grid_edge = grid_size / 40
    for grid_t in grids:
        #
        x_start = grid_t[0] - grid_size / 2 + grid_edge
        y_start = grid_t[1] - grid_size / 2 + grid_edge
        grid_width = grid_size - grid_edge * 2
        #
        color_t = [0.8, 0.8, 0.8, 0.5]
        #
        axs_fr.add_patch(pc.Rectangle((x_start, y_start), grid_width, grid_width, color=color_t))
        axs_bk.add_patch(pc.Rectangle((x_start, y_start), grid_width, grid_width, color=color_t))
        axs_lt.add_patch(pc.Rectangle((x_start, y_start), grid_width, grid_width, color=color_t))
        axs_rt.add_patch(pc.Rectangle((x_start, y_start), grid_width, grid_width, color=color_t))

    #
    color_0 = color_set[0]
    color_motion = [color_set[1], color_set[5], color_set[15]]
    color_action = color_set[9]

    #
    # FR actions
    initial_pos = [0, 0, 'E']
    draw_fuselage(axs_fr, [initial_pos[0], initial_pos[1]], initial_pos[2], face_color=color_0)
    tgt_prob_set = [(0.25, -0.25, 'E', 0.1),
                    (0.25,  0,    'E', 0.8),
                    (0.25,  0.25, 'E', 0.1),]
    text_xy_set = [( 0.275, -0.195),
                   ( 0.275,  0.055),
                   ( 0.275,  0.285),
                   (-0.245,  0.)]
    for i in range(0, tgt_prob_set.__len__()):
        #
        x_t_t = initial_pos[0]
        y_t_t = initial_pos[1]
        #
        x_next_t = tgt_prob_set[i][0]
        y_next_t = tgt_prob_set[i][1]
        heading_next_t = tgt_prob_set[i][2]
        probability_t  = tgt_prob_set[i][3]
        #
        axs_fr.plot([x_t_t, x_next_t], [y_t_t, y_next_t], color=color_motion[i], lw=5.5, linestyle='--', alpha=0.375)
        draw_fuselage(axs_fr, (x_next_t, y_next_t), heading_next_t, face_color=color_motion[i], alpha=0.45)
        #
        x_text_t = text_xy_set[i][0]
        y_text_t = text_xy_set[i][1]
        #
        text_t = axs_fr.text(x_text_t, y_text_t, r"$%.2f$" % (probability_t,), color=color_motion[i], fontsize=22.5)
    x_text_t = text_xy_set[list(text_xy_set).__len__() - 1][0]
    y_text_t = text_xy_set[list(text_xy_set).__len__() - 1][1]
    text_t = axs_fr.text(x_text_t, y_text_t, r"\textbf{FR}", color=color_action, fontsize=25)

    #
    # BK actions
    initial_pos = [0, 0, 'E']
    draw_fuselage(axs_bk, [initial_pos[0], initial_pos[1]], initial_pos[2], face_color=color_0)
    #
    tgt_prob_set = [(-0.25, -0.25, 'E', 0.15),
                    (-0.25, 0,     'E', 0.7),
                    (-0.25, 0.25,  'E', 0.15), ]
    text_xy_set = [(-0.225,  -0.315),
                   (-0.225,  -0.075),
                   (-0.225,   0.285),
                   ( 0.1475,  0.)]
    for i in range(0, tgt_prob_set.__len__()):
        #
        x_t_t = initial_pos[0]
        y_t_t = initial_pos[1]
        #
        x_next_t = tgt_prob_set[i][0]
        y_next_t = tgt_prob_set[i][1]
        heading_next_t = tgt_prob_set[i][2]
        probability_t = tgt_prob_set[i][3]
        #
        axs_bk.plot([x_t_t, x_next_t], [y_t_t, y_next_t], color=color_motion[i], lw=5.5, linestyle='--', alpha=0.375)
        draw_fuselage(axs_bk, (x_next_t, y_next_t), heading_next_t, face_color=color_motion[i], alpha=0.45)
        #
        x_text_t = text_xy_set[i][0]
        y_text_t = text_xy_set[i][1]
        #
        text_t = axs_bk.text(x_text_t, y_text_t, r"$%.2f$" % (probability_t,), color=color_motion[i], fontsize=22.5)
    x_text_t = text_xy_set[list(text_xy_set).__len__() - 1][0]
    y_text_t = text_xy_set[list(text_xy_set).__len__() - 1][1]
    text_t = axs_bk.text(x_text_t, y_text_t, r"\textbf{BK}", color=color_action, fontsize=25)

    #
    # LT actions
    initial_pos = [0, 0.0, 'E']
    draw_fuselage(axs_lt, [initial_pos[0], initial_pos[1]], initial_pos[2], face_color=color_0)
    #
    tgt_prob_set = [(0.05, 0,     'E', 0.05),
                    (0.05, 0.05,  'N', 0.9),
                    (0,    0.05,  'W', 0.05), ]
    text_xy_set = [(0.,     -0.115),                       # 不转
                   (-0.25,   0.0),                         # 转90
                   (0.,     0.15),                         # 转90, 再多转90
                   (0.15,  0)]
    alpha_set = [0.25, 0.45, 0.45]
    for i in range(0, tgt_prob_set.__len__()):
        #
        x_t_t = initial_pos[0]
        y_t_t = initial_pos[1]
        #
        x_next_t = tgt_prob_set[i][0]
        y_next_t = tgt_prob_set[i][1]
        heading_next_t = tgt_prob_set[i][2]
        probability_t = tgt_prob_set[i][3]
        #
        draw_fuselage(axs_lt, (x_next_t, y_next_t), heading_next_t, face_color=color_motion[i], alpha=alpha_set[i])
        #
        x_text_t = text_xy_set[i][0]
        y_text_t = text_xy_set[i][1]
        #
        text_t = axs_lt.text(x_text_t, y_text_t, r"$%.2f$" % (probability_t,), color=color_motion[i], fontsize=22.5)
    x_text_t = text_xy_set[list(text_xy_set).__len__() - 1][0]
    y_text_t = text_xy_set[list(text_xy_set).__len__() - 1][1]
    text_t = axs_lt.text(x_text_t, y_text_t, r"\textbf{TL}", color=color_action, fontsize=25)

    #
    # RT actions
    initial_pos = [0, 0.0, 'E']
    draw_fuselage(axs_rt, [initial_pos[0], initial_pos[1]], initial_pos[2], face_color=color_0)
    #
    tgt_prob_set = [(0.05, 0,      'E', 0.05),
                    (0.05, -0.05,  'S', 0.9),
                    (0,    -0.05,  'W', 0.05), ]
    text_xy_set = [(0.,     0.065),                      # 不转
                   (0.15,   0.0),                        # 转90
                   (0.,    -0.195),                      # 转90, 再多转90
                   (-0.25,  0)]
    alpha_set = [0.25, 0.45, 0.45]
    for i in range(0, tgt_prob_set.__len__()):
        #
        x_t_t = initial_pos[0]
        y_t_t = initial_pos[1]
        #
        x_next_t = tgt_prob_set[i][0]
        y_next_t = tgt_prob_set[i][1]
        heading_next_t = tgt_prob_set[i][2]
        probability_t = tgt_prob_set[i][3]
        #
        draw_fuselage(axs_rt, (x_next_t, y_next_t), heading_next_t, face_color=color_motion[i], alpha=alpha_set[i])
        #
        x_text_t = text_xy_set[i][0]
        y_text_t = text_xy_set[i][1]
        #
        text_t = axs_rt.text(x_text_t, y_text_t, r"$%.2f$" % (probability_t,), color=color_motion[i], fontsize=22.5)
    x_text_t = text_xy_set[list(text_xy_set).__len__() - 1][0]
    y_text_t = text_xy_set[list(text_xy_set).__len__() - 1][1]
    text_t = axs_rt.text(x_text_t, y_text_t, r"\textbf{TR}", color=color_action, fontsize=25)

    #
    axs_fr.set_xlim(-grid_size * 1.5, grid_size * 1.5)
    axs_bk.set_xlim(-grid_size * 1.5, grid_size * 1.5)
    axs_lt.set_xlim(-grid_size * 1.5, grid_size * 1.5)
    axs_rt.set_xlim(-grid_size * 1.5, grid_size * 1.5)
    #
    axs_fr.set_ylim(-grid_size * 1.5, grid_size * 1.5)
    axs_bk.set_ylim(-grid_size * 1.5, grid_size * 1.5)
    axs_lt.set_ylim(-grid_size * 1.5, grid_size * 1.5)
    axs_rt.set_ylim(-grid_size * 1.5, grid_size * 1.5)
    #
    axs_fr.xaxis.set_major_locator(ticker.MultipleLocator(grid_size))
    axs_bk.xaxis.set_major_locator(ticker.MultipleLocator(grid_size))
    axs_lt.xaxis.set_major_locator(ticker.MultipleLocator(grid_size))
    axs_rt.xaxis.set_major_locator(ticker.MultipleLocator(grid_size))
    #
    axs_fr.yaxis.set_major_locator(ticker.MultipleLocator(grid_size))
    axs_bk.yaxis.set_major_locator(ticker.MultipleLocator(grid_size))
    axs_lt.yaxis.set_major_locator(ticker.MultipleLocator(grid_size))
    axs_rt.yaxis.set_major_locator(ticker.MultipleLocator(grid_size))
    #
    axs_fr.set_xlabel(r'$y(m)$')
    axs_fr.set_ylabel(r'$x(m)$')
    axs_fr.set_aspect('equal')
    #
    axs_bk.set_xlabel(r'$y(m)$')
    axs_bk.set_ylabel(r'$x(m)$')
    axs_bk.set_aspect('equal')
    #
    axs_lt.set_xlabel(r'$y(m)$')
    axs_lt.set_ylabel(r'$x(m)$')
    axs_lt.set_aspect('equal')
    #
    axs_rt.set_xlabel(r'$y(m)$')
    axs_rt.set_ylabel(r'$x(m)$')
    axs_rt.set_aspect('equal')

    plt.tight_layout()  # 调整布局以避免子图之间重叠
