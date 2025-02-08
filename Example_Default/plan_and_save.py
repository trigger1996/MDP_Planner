from MDP_TG.mdp import Motion_MDP
from MDP_TG.dra import Dra, Product_Dra
#from MDP_TG.lp import syn_full_plan, syn_full_plan_rex
#from MDP_TG.vis import visualize_run                        # sudo apt install texlive-latex-extra dvipng -y
from User.lp import syn_full_plan, synthesize_full_plan_w_opacity
from User.dra2 import product_mdp2
from Map.vis_4_plan_and_save import visualize_run_sequence, visualize_trajectories, visualiza_in_animation
from Map.vis_4_plan_and_save import draw_mdp_principle, draw_action_principle
from User.vis2 import print_c

import pickle
import time
import networkx

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

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

def obtain_all_aps_from_mdp(mdp:Motion_MDP):
    ap_list = []
    for state_t in mdp.nodes():
        state_attr_t = mdp.nodes()[state_t]
        label_t = state_attr_t['label']
        for ap_t in list(state_attr_t['label'])[0]:
            ap_list.append(ap_t)

    return list(set(ap_list))

def plan_and_save_with_opacity(ws_robot_model, task, optimizing_ap, risk_pr, differential_exp_cost):

    #
    # 这里的MDP系统是将栅格和状态挂在一起
    # 相当于一个栅格对应多个状态，e.g., 同样位置车头不同朝向在以前算一个状态，在这里可以算4个
    print_c("[synthesize_w_opacity] process STARTED: risk: %f differential_exp_cost: %f" % (risk_pr, differential_exp_cost,), color=31)

    t0 = time.time()
    (robot_nodes, robot_edges, U, initial_node, initial_label) = ws_robot_model[:]
    motion_mdp = Motion_MDP(robot_nodes, robot_edges, U,
                            initial_node, initial_label)
    t2 = time.time()
    print('MDP done, time: %s' % str(t2-t0))

    ap_list = obtain_all_aps_from_mdp(motion_mdp)

    pickle.dump(networkx.get_edge_attributes(motion_mdp, 'prop'),
                open('motion_mdp_edges.p', "wb"))
    print('motion_mdp_edges.p saved!')

    # ----
    best_all_plan, prod_dra_pi = synthesize_full_plan_w_opacity(motion_mdp, task, optimizing_ap, ap_list, risk_pr, differential_exp_cost, is_enable_inter_state_constraints=False)


    # ----
    #
    # 在这个地方计算S_f, 即MEC
    # S_f 数据结构: [MEC, MEC ^ Ip, loop_act],
    #       第一项是所有MEC,
    #       第二项是所有MEC和Ip的交集,
    #       第三项是一个字典,
    #       for s in mdp.nodes():
    #           A[s] = mdp.nodes[s]['act'].copy()

    # Added
    total_T = 200
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
            X, L, U, M, PX = prod_dra_pi.execution(best_all_plan, total_T, state_seq, label_seq)

            XX.append(X)
            LL.append(L)
            UU.append(U)
            MM.append(M)
            PP.append(PX)

        print('[Product Dra] process all done')

        #fig = visualize_run_sequence(XX, LL, UU, MM, 'surv_result', is_visuaize=False)


    except:
        print_c("No best plan synthesized, try re-run this program", color=33)

    draw_mdp_principle()
    draw_action_principle()

    visualize_trajectories(motion_mdp, initial_node, XX, LL, UU, MM, 'surv_trajectories', is_gradient_color=True)
    visualiza_in_animation(motion_mdp, initial_node, XX, LL, UU, MM, 'surv_animation', is_show_action=True, is_gradient_color=True)
    plt.show()

if __name__ == "__main__":
    ws_robot_model = build_model()
    # ----
    # test_task = 'G F base1'
    # all_base = '& G F base1 & G F base2 & G F base3 G ! obstacle'
    # order1 = 'G i supply X U ! supply base'
    # order2 = 'G i base X U ! base supply'
    # order = '& %s %s' % (order1, order2)
    # task1 = '& %s & G ! obstacle %s' % (all_base, order2)
    # task2 = '& %s G F supply' % all_base
    # task3 = '& %s %s' % (all_base, order2)
    #
    # 改用标准的Ltl
    # https://spot.lre.epita.fr/ltlfilt.html#:~:text=ltlfilt.%20Table%20of%20Contents.%20Changing%20syntaxes.%20Altering%20the%20formula.%20Filtering.
    all_base = 'GF base1 & G !obstacle'
    #
    optimizing_ap = 'base3'
    risk_threshold = 0.05                                        # default:  0.1
    differential_exp_cost = 1.590106 - 0.075                     #           1.590106

    plan_and_save_with_opacity(ws_robot_model, all_base, optimizing_ap, risk_threshold, differential_exp_cost)

