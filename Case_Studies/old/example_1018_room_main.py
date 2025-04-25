import time
from subprocess import check_output
from Map.old.example_room_1018 import build_model
from MDP_TG.mdp import Motion_MDP
from MDP_TG.dra import Dra, Product_Dra
from MDP_TG.lp import syn_full_plan
from User.vis2 import print_c


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


def room_example_main():

    ltl_formula = 'GF (gather -> (!gather U drop))'
    ltl_formula_converted = ltl_convert(ltl_formula)

    robot_nodes, robot_edges, U, initial_node, initial_label = build_model()
    motion_mdp = Motion_MDP(robot_nodes, robot_edges, U,
                            initial_node, initial_label)

    dra = Dra(ltl_formula_converted)

    # ----
    prod_dra = Product_Dra(motion_mdp, dra)
    # prod_dra.dotify()

    # ----
    #
    # 在这个地方计算S_f, 即MEC
    # S_f 数据结构: [MEC, MEC ^ Ip, loop_act],
    #       第一项是所有MEC,
    #       第二项是所有MEC和Ip的交集,
    #       第三项是一个字典,
    #       for s in mdp.nodes():
    #           A[s] = mdp.nodes[s]['act'].copy()
    # prod_dra.compute_S_f_rex()
    prod_dra.compute_S_f()
    t42 = time.time()

    # ------
    gamma = 0.1
    d = 100
    # best_all_plan = syn_full_plan_rex(prod_dra, gamma, d)
    best_all_plan = syn_full_plan(prod_dra, gamma)

    # Added
    # for printing policies
    print_c("state action: probabilities")
    print_c("Prefix", color=42)
    #
    state_in_prefix = [ state_t for state_t in best_all_plan[0][0] ]
    #state_in_prefix.sort(key=cmp_to_key(sort_grids))
    #for state_t in best_all_plan[0][0]:
    for state_t in state_in_prefix:
        print_c("%s, %s: %s" % (str(state_t), str(best_all_plan[0][0][state_t][0]), str(best_all_plan[0][0][state_t][1]), ), color=42)
    #
    print_c("Suffix", color=45)
    state_in_suffix = [ state_t for state_t in best_all_plan[1][0] ]
    #state_in_suffix.sort(key=cmp_to_key(sort_grids))
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
            #
            # Added for comparison
            # Y = run_2_observations_seqs(X_U)
            # X_INV, AP_INV = observation_seq_2_inference(Y)
            # print_c(X_U, color=color_init)
            # print_c(Y, color=color_init)
            # print_c(X_INV, color=color_init)
            # print_c(AP_INV, color=color_init)
            #
            color_init += 1
        #fig = visualize_run_sequence(XX, LL, UU, MM, 'surv_result', is_visuaize=False)


    except:
        print_c("No best plan synthesized, try re-run this program", color=33)

    #draw_action_principle()
    #draw_mdp_principle()



if __name__ == "__main__":
    room_example_main()
