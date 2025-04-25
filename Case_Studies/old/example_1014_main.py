from MDP_TG.mdp import Motion_MDP
from MDP_TG.dra import Dra, Product_Dra
from MDP_TG.lp import syn_full_plan
from Map.old.example_8x5_1014 import build_model
from Map.old.example_8x5_1014 import visualize_trajectories, visualiza_in_animation
from User.vis2 import print_c

from subprocess import check_output
from functools import cmp_to_key
from User.grid_utils import sort_grids

import pickle
import time
import networkx

import matplotlib.pyplot as plt
plt.use("TkAgg")

def ltl_convert(task, is_display=True):
    #
    # https://www.ltl2dstar.de/docs/ltl2dstar.html#:~:text=ltl2dstar%20is%20designed%20to%20use%20an%20external%20tool%20to%20convert
    # LTL是有两个格式的, 一个ltl2dstar notation, 一个spin notation
    # 后者是常用的, 要从后者转到前者ltl2dstar才能用
    cmd_ltl_convert = 'ltlfilt -l -f \'%s\'' % (task, )
    ltl_converted = str(check_output(cmd_ltl_convert, shell=True))
    ltl_converted = ltl_converted[2 : len(ltl_converted) - 3]               # tested
    if is_display:
        print_c('converted ltl: ' + ltl_converted)

    return ltl_converted


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
                open('motion_mdp_edges_8x5_1014.p', "wb"))
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
    #
    # 在这个地方计算S_f, 即MEC
    # S_f 数据结构: [MEC, MEC ^ Ip, loop_act],
    #       第一项是所有MEC,
    #       第二项是所有MEC和Ip的交集,
    #       第三项是一个字典,
    #       for s in mdp.nodes():
    #           A[s] = mdp.nodes[s]['act'].copy()
    #prod_dra.compute_S_f_rex()
    prod_dra.compute_S_f()
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
    # for printing policies
    print_c("state action: probabilities")
    print_c("Prefix", color=42)
    #
    state_in_prefix = [ state_t for state_t in best_all_plan[0][0] ]
    state_in_prefix.sort(key=cmp_to_key(sort_grids))
    #for state_t in best_all_plan[0][0]:
    for state_t in state_in_prefix:
        print_c("%s, %s: %s" % (str(state_t), str(best_all_plan[0][0][state_t][0]), str(best_all_plan[0][0][state_t][1]), ), color=42)
    #
    print_c("Suffix", color=45)
    state_in_suffix = [ state_t for state_t in best_all_plan[1][0] ]
    state_in_suffix.sort(key=cmp_to_key(sort_grids))
    #for state_t in best_all_plan[1][0]:
    for state_t in state_in_suffix:
        print_c("%s, %s: %s" % (str(state_t), str(best_all_plan[1][0][state_t][0]), str(best_all_plan[1][0][state_t][1]), ), color=45)

    # for visualization
    total_T = 200
    state_seq = [ initial_node, ]
    label_seq = [ initial_label, ]
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

        color_init = 32
        for i in range(0, XX.__len__()):
            X_U = []
            for j in range(0, XX[i].__len__()):
                X_U.append(XX[i][j])
                X_U.append(UU[i][j])
            print_c(X_U, color=color_init)
            color_init += 1
        #fig = visualize_run_sequence(XX, LL, UU, MM, 'surv_result', is_visuaize=False)


    except:
        print_c("No best plan synthesized, try re-run this program", color=33)

    #draw_action_principle()
    #draw_mdp_principle()


    visualize_trajectories(motion_mdp, initial_node, XX, LL, UU, MM, 'surv_trajectories', is_gradient_color=True)
    visualiza_in_animation(motion_mdp, initial_node, XX, LL, UU, MM, 'surv_animation', is_show_action=True, is_gradient_color=True)
    plt.show()

if __name__ == "__main__":
    ws_robot_model = build_model()

    # ----
    ltl_formula =  'GF (supply -> drop) && G !obstacle'          # '& G F supply & G F drop G ! obstacle'
    ltl_formula_converted = ltl_convert(ltl_formula)
    plan_and_save(ws_robot_model, ltl_formula_converted)

