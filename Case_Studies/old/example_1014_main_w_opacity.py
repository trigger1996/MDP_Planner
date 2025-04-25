from MDP_TG.mdp import Motion_MDP
#from MDP_TG.lp import syn_full_plan, syn_full_plan_rex
#from MDP_TG.vis import visualize_run                        # sudo apt install texlive-latex-extra dvipng -y
from User.lp import synthesize_full_plan_w_opacity
from Map.old.example_8x5_1014 import build_model
from Map.old.example_8x5_1014 import visualize_trajectories, visualiza_in_animation
from Map.old.example_8x5_1014 import draw_mdp_principle, draw_action_principle
from User.vis2 import print_c
from Map.old.example_8x5_1014 import observation_func as observation_func_1014

import pickle
import time
import networkx

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


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
    best_all_plan, prod_dra_pi = synthesize_full_plan_w_opacity(motion_mdp, task, optimizing_ap, ap_list, risk_pr, differential_exp_cost, is_enable_inter_state_constraints=False, observation_func=observation_func_1014)


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

    #try:
    if True:
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


    #except:
    #    print_c("No best plan synthesized, try re-run this program", color=33)

    draw_mdp_principle()
    draw_action_principle()

    visualize_trajectories(motion_mdp, initial_node, XX, LL, UU, MM, 'surv_trajectories', is_gradient_color=True)
    visualiza_in_animation(motion_mdp, initial_node, XX, LL, UU, MM, 'surv_animation', is_show_action=True, is_gradient_color=True)
    plt.show()

if __name__ == "__main__":
    ws_robot_model = build_model(is_enable_single_direction=True)

    #
    # 改用标准的Ltl
    # https://spot.lre.epita.fr/ltlfilt.html#:~:text=ltlfilt.%20Table%20of%20Contents.%20Changing%20syntaxes.%20Altering%20the%20formula.%20Filtering.
    ltl_formula =  'GF (supply -> drop)'          # '& G F supply & G F drop G ! obstacle'
    #
    optimizing_ap = 'drop'
    risk_threshold = 0.05                                        # default:  0.1
    differential_exp_cost = 1.590106 - 0.075                     #           1.590106

    plan_and_save_with_opacity(ws_robot_model, ltl_formula, optimizing_ap, risk_threshold, differential_exp_cost)

