from MDP_TG.mdp import Motion_MDP
from MDP_TG.dra import Dra, Product_Dra
from MDP_TG.lp import syn_full_plan, syn_full_plan_rex
#from MDP_TG.vis import visualize_run                        # sudo apt install texlive-latex-extra -y

import pickle
import time
import networkx

import matplotlib
import matplotlib.pyplot as plt


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


def build_model(N_x=8, N_y=10):
    t0 = time.time()

    # -------- exp -------
    WS_d = 0.25
    WS_node_dict = {
        # base stations
        ((2*N_x-1)*WS_d, WS_d): {frozenset(['base1', 'base']): 1.0,
                               frozenset(): 0.0, },
        ((2*N_x-1)*WS_d, (2*N_y-1)*WS_d): {frozenset(['base2', 'base']): 1.0,
                                       frozenset(): 0.0, },
        (WS_d, (2*N_y-1)*WS_d): {frozenset(['base3', 'base']): 1.0,
                               frozenset(): 0.0, },
        (N_x*WS_d, N_y*WS_d): {frozenset(['supply']): 1.0,
                           frozenset(): 0.0, },  # high
    }

    # add high prob obstacles
    # for k_x in range(1, 7, 2):
    #     WS_node_dict[(k_x*WS_d, 11*WS_d)
    #                  ] = {frozenset(['obstacle', 'top']): 1.0, frozenset(): 0.0, }

    for x in range(0, N_x):
        for y in range(0, N_y):
            node = ((2*x+1)*WS_d, (2*y+1)*WS_d)
            if node not in WS_node_dict:
                WS_node_dict[node] = {frozenset(): 1.0, }

    print('WS_node_dict_size', len(WS_node_dict))

    pickle.dump((WS_node_dict, WS_d), open('ws_model.p', "wb"))
    print('ws_model.p saved!')
    # ----
    # visualize_world(WS_d, WS_node_dict, 'world')
    # t1 = time.time()
    # print 'visualize world done, time: %s' %str(t1-t0)
    # ------------------------------------
    robot_nodes = dict()
    for loc, prop in WS_node_dict.items():
        for d in ['N', 'S', 'E', 'W']:
            robot_nodes[(loc[0], loc[1], d)] = prop
    # ------------------------------------
    initial_node = (3*WS_d, 3*WS_d, 'E')
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
            t_nodes = [(fx-2*WS_d, fy+2*WS_d, fd), (fx, fy+2*WS_d, fd),
                       (fx+2*WS_d, fy+2*WS_d, fd)]
        if fd == 'S':
            t_nodes = [(fx-2*WS_d, fy-2*WS_d, fd), (fx, fy-2*WS_d, fd),
                       (fx+2*WS_d, fy-2*WS_d, fd)]
        if fd == 'E':
            t_nodes = [(fx+2*WS_d, fy-2*WS_d, fd), (fx+2*WS_d, fy, fd),
                       (fx+2*WS_d, fy+2*WS_d, fd)]
        if fd == 'W':
            t_nodes = [(fx-2*WS_d, fy-2*WS_d, fd), (fx-2*WS_d, fy, fd),
                       (fx-2*WS_d, fy+2*WS_d, fd)]
        for k, tnode in enumerate(t_nodes):
            if tnode in list(robot_nodes.keys()):
                robot_edges[(fnode, u, tnode)] = (P_FR[k], c)
        # action BK
        u = U[1]
        c = C[1]
        if fd == 'N':
            t_nodes = [(fx-2*WS_d, fy-2*WS_d, fd), (fx, fy-2*WS_d, fd),
                       (fx+2*WS_d, fy-2*WS_d, fd)]
        if fd == 'S':
            t_nodes = [(fx-2*WS_d, fy+2*WS_d, fd), (fx, fy+2*WS_d, fd),
                       (fx+2*WS_d, fy+2*WS_d, fd)]
        if fd == 'E':
            t_nodes = [(fx-2*WS_d, fy-2*WS_d, fd), (fx-2*WS_d, fy, fd),
                       (fx-2*WS_d, fy+2*WS_d, fd)]
        if fd == 'W':
            t_nodes = [(fx+2*WS_d, fy-2*WS_d, fd), (fx+2*WS_d, fy, fd),
                       (fx+2*WS_d, fy+2*WS_d, fd)]
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
    # ----
    ws_robot_model = (robot_nodes, robot_edges, U,
                      initial_node, initial_label)
    t1 = time.time()
    print('ws_robot_model returned, time: %s' % str(t1-t0))
    return ws_robot_model

def visualize_run(XX, LL, UU, MM, name=None):
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

    plt.show()

    return figure

def plan_and_save(ws_robot_model, task):

    #
    # 这里的MDP系统是将栅格和状态挂在一起
    # 相当于一个栅格对应多个状态，e.g., 同样位置车头不同朝向在以前算一个状态，在这里可以算4个

    t0 = time.time()
    (robot_nodes, robot_edges, U, initial_node, initial_label) = ws_robot_model[:]
    motion_mdp = Motion_MDP(robot_nodes, robot_edges, U,
                            initial_node, initial_label)
    t2 = time.time()
    print('MDP done, time: %s' % str(t2-t0))

    pickle.dump(networkx.get_edge_attributes(motion_mdp, 'prop'),
                open('motion_mdp_edges.p', "wb"))
    print('motion_mdp_edges.p saved!')
    # ----
    dra = Dra(task)
    t3 = time.time()
    print('DRA done, time: %s' % str(t3-t2))

    # ----
    prod_dra = Product_Dra(motion_mdp, dra)
    # prod_dra.dotify()
    t41 = time.time()
    print('Product DRA done, time: %s' % str(t41-t3))

    pickle.dump((networkx.get_edge_attributes(prod_dra, 'prop'),
                prod_dra.graph['initial']), open('prod_dra_edges.p', "wb"))
    print('prod_dra_edges.p saved')

    # ----
    prod_dra.compute_S_f_rex()
    t42 = time.time()
    print('Compute ASCC done, time: %s' % str(t42-t41))

    # ------
    gamma = 0.1
    d = 100
    #best_all_plan = syn_full_plan_rex(prod_dra, gamma, d)
    best_all_plan = syn_full_plan(prod_dra, gamma)
    t5 = time.time()
    print('Plan synthesis done, time: %s' % str(t5-t42))

    pickle.dump(best_all_plan, open('best_plan.p', "wb"))
    print('best_plan.p saved!')

    # Added
    total_T = 20
    state_seq = [initial_node, ]
    label_seq = [initial_label, ]
    N = 5

    try:
        XX = []
        LL = []
        UU = []
        MM = []
        PP = []
        for n in range(0, N):
            X, L, U, M, PX = prod_dra.execution(best_all_plan, total_T, state_seq, label_seq)

            XX.append(X)
            LL.append(L)
            UU.append(U)
            MM.append(M)
            PP.append(PX)

        print('[Product Dra] process all done')

        fig = visualize_run(XX, LL, UU, MM, 'surv_result')

    except:
        print_c("No best plan synthesized, try re-run this program", color=33)

if __name__ == "__main__":
    ws_robot_model = build_model()
    # ----
    test_task = 'G F base1'
    all_base = '& G F base1 & G F base2 & G F base3 G ! obstacle'
    order1 = 'G i supply X U ! supply base'
    order2 = 'G i base X U ! base supply'
    order = '& %s %s' % (order1, order2)
    task1 = '& %s & G ! obstacle %s' % (all_base, order2)
    task2 = '& %s G F supply' % all_base
    task3 = '& %s %s' % (all_base, order2)
    plan_and_save(ws_robot_model, all_base)
