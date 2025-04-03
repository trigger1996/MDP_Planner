from math import fabs, sqrt

import networkx as nx
from networkx import DiGraph, is_connected
from MDP_TG.mdp import find_MECs, find_SCCs
from MDP_TG.dra import Product_Dra
from User.vis2 import print_c

import random


def observation_func_1(state, observation='X'):
    if observation == 'X':
        return state[0][0]
    if observation == 'y':
        return state[0][1]     # x and y are NOT inverted here, they are only inverted in plotting

def observation_func_2(state):
    observation_set = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                       (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                       (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
                       (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
                       (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
                       (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),]
    min_dist = 1e6
    observed_state = (0, 0)
    for state_obs_t in observation_set:
        x_obs = state_obs_t[0]
        y_obs = state_obs_t[1]
        #
        dx = state[0][0] - x_obs
        dy = state[0][1] - y_obs
        dist = sqrt(dx * dx + dy * dy)
        if dist < min_dist:
            min_dist = dist
            observed_state = state_obs_t
    return observed_state

def is_with_identical_observation_1(state, state_prime):
    #
    # the observation is X coordinate of the state
    if state[0][0] == state_prime[0][0]:
        return True
    else:
        return False

def is_with_identical_observation_2(state, state_prime):
    #
    # the observation is Y coordinate of the state
    if state[0][1] == state_prime[0][1]:
        return True
    else:
        return False

def is_with_identical_observation_3(state, state_prime, dist_threshold=0.5):
    #
    # if the Cartesian distance is smaller than the given threshold
    dx = state[0][0] - state_prime[0][0]
    dy = state[0][1] - state_prime[0][1]
    dist = sqrt(dx * dx + dy * dy)
    if dist <= dist_threshold:
        return True
    else:
        return False

def is_with_identical_observation_4(state, state_prime, dist_threshold=1.):
    #
    # if the Manhattan distance is smaller than the given threshold
    dx = state[0][0] - state_prime[0][0]
    dy = state[0][1] - state_prime[0][1]
    dist = fabs(dx) + fabs(dy)
    if dist <= dist_threshold:
        return True
    else:
        return False

def is_ap_identical(state_pi, state_gamma):
    if list(state_pi[1]).__len__() == 0 and list(state_gamma[1]).__len__() == 0:
        return True
    elif set(state_pi[1]) == set(state_gamma[1]):
        return True
    else:
        return False


def find_initial_state(y_in_sf_pi, y_in_sf_gamma, state_set_gamma, observation_func=observation_func_1):
    state_list = []
    for state_pi_t in y_in_sf_pi.keys():
        # TODO
        # to explain reasons
        # if y_in_sf_pi[state_pi_t] == 0:
        #     continue
        for state_gamma_t in y_in_sf_gamma:
            # if y_in_sf_gamma[state_gamma_t] == 0:
            #     continue
            #
            if observation_func(state_pi_t) == observation_func(state_gamma_t):
                if is_ap_identical(state_pi_t, state_gamma_t):
                    state_list.append((state_pi_t, state_gamma_t, ))
    return state_list

def obtain_differential_expected_cost(current_action, edge_pi, edge_gamma):
    #
    pr_pi   = edge_pi[2]['prop'][current_action][0]
    cost_pi = edge_pi[2]['prop'][current_action][1]
    expected_cost_pi = pr_pi * cost_pi
    #
    try:
        pr_gamma   = edge_gamma[2]['prop'][current_action][0]
        cost_gamma = edge_gamma[2]['prop'][current_action][1]
        expected_cost_gamma = pr_gamma * cost_gamma
        #
        different_expected_cost = fabs(expected_cost_pi - expected_cost_gamma)
    except KeyError:
        #
        # TODO
        # 看来两个必须action相同
        # 不然根据MDP的定义, transition cost是X \times U \times -> R, U不同整个transition也不同, 最后秋出来的cost也不同
        # print_c("[synthesize_w_opacity] WARNING no corresponding actions  %s // %s" % (str(edge_pi), str(edge_gamma),), color=36)
        different_expected_cost = fabs(expected_cost_pi)

    return different_expected_cost

def sort_scc_list(x):
    return x.__len__()

def act_by_plan(prod_mdp, best_plan, prod_state):
    # ----choose the randomized action by the optimal policy----
    # recall that {best_plan = [plan_prefix, prefix_cost, prefix_risk, y_in_sf],
    # [plan_suffix, suffix_cost, suffix_risk], [MEC[0], MEC[1], Sr, Sd], plan_bad]}
    plan_prefix = best_plan[0][0]
    plan_suffix = best_plan[1][0]
    plan_bad = best_plan[3]
    if (prod_state in plan_prefix):
        # print 'In prefix'
        U = plan_prefix[prod_state][0]
        P = plan_prefix[prod_state][1]
        rdn = random.random()
        pc = 0
        for k, p in enumerate(P):
            pc += p
            if pc > rdn:
                break
        # print('%s action chosen: %s' % (str(prod_state), str(U[k], )))
        return U[k], 0
    elif (prod_state in plan_suffix):
        # print 'In suffix'
        U = plan_suffix[prod_state][0]
        P = plan_suffix[prod_state][1]
        rdn = random.random()
        pc = 0
        for k, p in enumerate(P):
            pc += p
            if pc > rdn:
                break
        # print('%s action chosen: %s' % (str(prod_state), str(U[k], )))
        if prod_state in best_plan[2][1]:
            return  U[k], 10                    # it is strange for best_plan[2][1] is for state set I_p, i.e., the states that Ap is satisfied
            #return U[k], 1
        else:
            return U[k], 1
    elif (prod_state in plan_bad):
        # print 'In bad states'
        U = plan_bad[prod_state][0]
        P = plan_bad[prod_state][1]
        rdn = random.random()
        pc = 0
        for k, p in enumerate(P):
            pc += p
            if pc > rdn:
                break
        # print 'action chosen: %s' %str(U[k])
        return U[k], 2
    else:
        print_c("Warning, current state %s is outside prefix and suffix !"  % (str(prod_state), ), color=33)
        return None, 4

class product_mdp2(Product_Dra):
    def __init__(self, mdp, dra):
        Product_Dra.__init__(self, mdp, dra)
        self.compute_S_f()
        #
        self.sync_amec_set = list()
        self.current_sync_amec_index = 0

    def compute_S_f(self):
        # ----find all accepting End components----
        S = set(self.nodes())
        acc_pairs = self.graph['accept']
        S_f = []
        k = 1
        for pair in acc_pairs:
            # ---for each accepting pair
            print("+++++++++++++++++++++++++++++++++++++")
            print("++++++++++++ acc_pair %s ++++++++++++" % k)
            print("+++++++++++++++++++++++++++++++++++++")
            S_fi = []
            Ip = pair[0]
            Hp = pair[1]
            print("Ip size: %s" % len(Ip))
            print("Hp size: %s" % len(Hp))
            # ---find all MECs
            MEC, Act = find_MECs(self, S.difference(Hp))
            # ---find accepting ones
            for T in MEC:
                common = set(T.intersection(Ip))
                if common:
                    if len(T) > 1:
                        S_fi.append([T, common, Act])
                        print('S_fii added to S_fi!!, size: %s' % len(T))
                    if len(T) == 1:  # self-loop
                        common_cp = common.copy()
                        s = common_cp.pop()
                        if self.has_edge(s, s):                             # Added
                            loop_act_set = set(self[s][s]['prop'].keys())
                            loop_act = dict()
                            loop_act[s] = loop_act_set
                            S_fi.append([T, common, loop_act])
                            print('S_fii added to S_fi!!, size: %s' % len(T))
            if len(S_fi) > 0:
                S_f.append(S_fi)
                print("****S_fi added to S_f!!!, size: %s******" % len(S_fi))
            k += 1
        self.Sf = S_f
        if S_f:
            print("-------Accepting MEC for Prod DRA Computed-------")
            print("acc_pair number: %s" % str(k-1))
            print("Sf AMEC number: %s" % len(S_f))
        else:
            print("No accepting ECs found!")
            print("Check your MDP and Task formulation")
            print("Or try the relaxed plan")

    def re_synthesize_sync_amec(self, y_in_sf_pi, y_in_sf_gamma, ap_pi, ap_gamma, MEC_pi, MEC_gamma, product_mdp_gamma:Product_Dra, observation_func=observation_func_2, is_re_compute_Sf=True):
        # amec:
        #   [0] amec
        #   [1] amec ^ Ip
        #   [2] action set
        #
        mec_state_set_pi    = MEC_pi[0]
        mec_state_set_gamma = MEC_gamma[0]

        stack_t = find_initial_state(y_in_sf_pi, y_in_sf_gamma, list(MEC_gamma[0]), observation_func=observation_func)
        stack_t = list(set(stack_t))
        visited = set()

        sync_mec_t = DiGraph()
        # for state_t in stack_t:
        #     sync_mec_t.add_node(state_t)
        while stack_t.__len__():
            current_state = stack_t.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            #
            next_state_list_pi    = list(self.out_edges(current_state[0], data=True))
            next_state_list_gamma = list(product_mdp_gamma.out_edges(current_state[1], data=True))
            #
            for edge_t_pi in next_state_list_pi:
                for edge_t_gamma in next_state_list_gamma:
                    #
                    next_state_pi    = edge_t_pi[1]
                    next_state_gamma = edge_t_gamma[1]
                    next_sync_state = (next_state_pi, next_state_gamma)
                    #
                    u_pi    = list(edge_t_pi[2]['prop'].keys())[0]
                    u_gamma = list(edge_t_gamma[2]['prop'].keys())[0]
                    #
                    if u_pi != u_gamma:
                        continue
                    #
                    is_next_state_pi_in_amec    = next_state_pi    in mec_state_set_pi
                    is_next_state_gamma_in_amec = next_state_gamma in mec_state_set_gamma
                    if not is_next_state_pi_in_amec or not is_next_state_gamma_in_amec:
                        continue
                    #
                    if observation_func(next_state_pi) == observation_func(next_state_gamma):
                        # TODO
                        # it seems that if the AP are required identical, then the AP-none_AP states will NOT taken into consideration
                        # in other words, the successive states will be synchronized by AP-AP states
                        #
                        # ON THE OTHER HAND
                        # on the aspect of the intruders, the only information he can receive is the observation sequences
                        # so the sync-MEC (, or the observer) should focus only on the observation sequences
                        # NOT the aps
                        #
                        #if is_ap_identical(next_state_pi, next_state_gamma):
                        if True:

                            #
                            trans_pr_cost_list = dict()
                            diff_expected_cost_list = dict()
                            for current_action_t in edge_t_pi[2]['prop'].keys():
                                #
                                transition_prop = edge_t_pi[2]['prop'][current_action_t][0]
                                transition_cost = edge_t_pi[2]['prop'][current_action_t][1]
                                diff_expected_cost = obtain_differential_expected_cost(current_action_t, edge_t_pi, edge_t_gamma)
                                #
                                trans_pr_cost_list[current_action_t] = (transition_prop, transition_cost, )
                                diff_expected_cost_list[current_action_t] = diff_expected_cost
                            #
                            sync_mec_t.add_edge(current_state, next_sync_state, prop=trans_pr_cost_list, diff_exp=diff_expected_cost_list)
                            stack_t.append(next_sync_state)
        #
        print_c("[synthesize_w_opacity] DFS completed, states: %d, edges: %d" % (sync_mec_t.nodes.__len__(), sync_mec_t.edges.__len__(),), color=33)
        if sync_mec_t.edges().__len__():
            #
            # TODO
            # 检查连接性
            scc_list = list(nx.strongly_connected_components(sync_mec_t))
            scc_list.sort(key=sort_scc_list,reverse=True)
            if scc_list.__len__() == 0:
                print_c("[synthesize_w_opacity] NO SCCs found ..." , color=33)

            num_node_removed = 0
            for i, state_list_t in enumerate(scc_list):
                if i == 0:
                    continue
                for state_t in state_list_t:
                    try:
                        sync_mec_t.remove_node(state_t)
                        num_node_removed += 1
                        print_c("[synthesize_w_opacity] removing node: " + str(state_t), color=35)
                    except:
                        pass

            print_c("[synthesize_w_opacity] number of state to remove: %d" % (num_node_removed,), color=33)

            #
            # for debugging
            # state_pi_list_in_amec = []
            # for sync_state_t in (sync_mec_t.nodes()):
            #     state_pi_list_in_amec.append(sync_state_t[0])
            # #
            # state_pi_list_not_in_amec = []
            # for state_pi_t in MEC_pi[0]:
            #     if state_pi_t not in state_pi_list_in_amec:
            #         state_pi_list_not_in_amec.append(state_pi_t)

            #
            self.sync_amec_set.append(sync_mec_t)
            self.current_sync_amec_index = self.sync_amec_set.__len__() - 1
        #
        print_c("[synthesize_w_opacity] Generated sync_amec, states: %d, edges: %d" % (sync_mec_t.nodes.__len__(), sync_mec_t.edges.__len__(),))

    def re_synthesize_sync_amec_rex(self, y_in_sf_pi, y_in_sf_gamma, ap_pi, ap_gamma, MEC_pi, MEC_gamma, product_mdp_gamma:Product_Dra, observation_func=observation_func_2, is_re_compute_Sf=True):
        # amec:
        #   [0] amec
        #   [1] amec ^ Ip
        #   [2] action set
        #
        mec_state_set_pi    = MEC_pi[0]
        mec_state_set_gamma = MEC_gamma[0]

        stack_t = find_initial_state(y_in_sf_pi, y_in_sf_gamma, list(MEC_gamma[0]), observation_func=observation_func)
        stack_t = list(set(stack_t))
        visited = set()

        sync_mec_t = DiGraph()
        # for state_t in stack_t:
        #     sync_mec_t.add_node(state_t)
        while stack_t.__len__():
            current_state = stack_t.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            #
            next_state_list_pi    = list(self.out_edges(current_state[0], data=True))
            next_state_list_gamma = list(product_mdp_gamma.out_edges(current_state[1], data=True))
            #
            for edge_t_pi in next_state_list_pi:
                for edge_t_gamma in next_state_list_gamma:
                    #
                    next_state_pi    = edge_t_pi[1]
                    next_state_gamma = edge_t_gamma[1]
                    next_sync_state = (next_state_pi, next_state_gamma)
                    #
                    u_pi    = list(edge_t_pi[2]['prop'].keys())[0]
                    u_gamma = list(edge_t_gamma[2]['prop'].keys())[0]
                    #
                    if u_pi != u_gamma:
                        continue
                    #
                    is_next_state_pi_in_amec    = next_state_pi    in mec_state_set_pi
                    is_next_state_gamma_in_amec = next_state_gamma in mec_state_set_gamma
                    if not is_next_state_pi_in_amec or not is_next_state_gamma_in_amec:
                        continue

                    # forall ap in pi
                    # exists ap in gamma
                    # TODO
                    # 到底是建立过程中删除, 还是建立后统一删除
                    # ap_list_t = list(next_state_pi[1])
                    # if ap_list_t.__len__() == 0:
                    #     ap_t = None
                    # else:
                    #     ap_t = list(next_state_pi[1])[0]
                    # ap_list_p_t = list(next_state_gamma[1])
                    # if ap_list_p_t.__len__() == 0:
                    #     ap_p_t = None
                    # else:
                    #     ap_p_t = list(next_state_gamma[1])[0]
                    # if ap_t == ap_pi:
                    #     if not ap_p_t == ap_gamma:
                    #         continue

                    #
                    # if observation_func(next_state_pi) == observation_func(next_state_gamma):         # moved to below
                    if True:
                        # TODO
                        # it seems that if the AP are required identical, then the AP-none_AP states will NOT taken into consideration
                        # in other words, the successive states will be synchronized by AP-AP states
                        #
                        # ON THE OTHER HAND
                        # on the aspect of the intruders, the only information he can receive is the observation sequences
                        # so the sync-MEC (, or the observer) should focus only on the observation sequences
                        # NOT the aps
                        #
                        # 2025.1.29
                        # Trick:
                        # Use small probability to maintain connectivity
                        #
                        #if is_ap_identical(next_state_pi, next_state_gamma):
                        if True:

                            #
                            trans_pr_cost_list = dict()
                            diff_expected_cost_list = dict()
                            for current_action_t in edge_t_pi[2]['prop'].keys():
                                #
                                transition_prop = edge_t_pi[2]['prop'][current_action_t][0]
                                transition_cost = edge_t_pi[2]['prop'][current_action_t][1]
                                if observation_func(next_state_pi) == observation_func(next_state_gamma):
                                    diff_expected_cost = obtain_differential_expected_cost(current_action_t, edge_t_pi, edge_t_gamma)
                                else:
                                    #
                                    # maintain connectivity
                                    diff_expected_cost = 1.e6
                                    #
                                trans_pr_cost_list[current_action_t] = (transition_prop, transition_cost, )
                                diff_expected_cost_list[current_action_t] = diff_expected_cost

                            #
                            sync_mec_t.add_edge(current_state, next_sync_state, prop=trans_pr_cost_list, diff_exp=diff_expected_cost_list)
                            stack_t.append(next_sync_state)
        #
        print_c("[synthesize_w_opacity] DFS completed, states: %d, edges: %d" % (sync_mec_t.nodes.__len__(), sync_mec_t.edges.__len__(),), color=33)
        if sync_mec_t.edges().__len__():
            #
            # 检查连接性
            scc_list = list(nx.strongly_connected_components(sync_mec_t))
            scc_list.sort(key=sort_scc_list,reverse=True)
            if scc_list.__len__() == 0:
                print_c("[synthesize_w_opacity] NO SCCs found ..." , color=33)

            num_node_removed = 0
            for i, state_list_t in enumerate(scc_list):
                if i == 0:
                    continue
                for state_t in state_list_t:
                    try:
                        sync_mec_t.remove_node(state_t)
                        num_node_removed += 1
                        print_c("[synthesize_w_opacity] removing node: " + str(state_t), color=35)
                    except:
                        pass

            # 然后去除其中不满足观测条件的点
            node_to_remove = []
            for node_t in sync_mec_t:
                ap_list_t = list(node_t[0][1])
                if ap_list_t.__len__() == 0:
                    ap_t = None
                else:
                    ap_t = list(node_t[0][1])[0]
                ap_list_p_t = list(node_t[1][1])
                if ap_list_p_t.__len__() == 0:
                    ap_p_t = None
                else:
                    ap_p_t = list(node_t[1][1])[0]
                if ap_t == ap_pi:
                    if not ap_p_t == ap_gamma:
                        node_to_remove.append(node_t)
            for node_t in node_to_remove:
                try:
                    sync_mec_t.remove_node(node_t)
                    print_c("[synthesize_w_opacity] removing node for inconsistent ap: " + str(node_t), color=35)
                except:
                    pass

            print_c("[synthesize_w_opacity] number of state to remove: %d" % (num_node_removed,), color=33)

            #
            # for debugging
            # state_pi_list_in_amec = []
            # for sync_state_t in (sync_mec_t.nodes()):
            #     state_pi_list_in_amec.append(sync_state_t[0])
            # #
            # state_pi_list_not_in_amec = []
            # for state_pi_t in MEC_pi[0]:
            #     if state_pi_t not in state_pi_list_in_amec:
            #         state_pi_list_not_in_amec.append(state_pi_t)

            #
            self.sync_amec_set.append(sync_mec_t)
            self.current_sync_amec_index = self.sync_amec_set.__len__() - 1
        #
        print_c("[synthesize_w_opacity] Generated sync_amec, states: %d, edges: %d" % (sync_mec_t.nodes.__len__(), sync_mec_t.edges.__len__(),))


    def execution(self, best_all_plan, total_T, state_seq, label_seq):
        # ----plan execution with or without given observation----
        t = 0
        X = []
        L = []
        U = []
        M = []
        PX = []
        m = 0
        # ----
        while (t <= total_T):
            if (t == 0):
                # print '---initial run----'
                mdp_state = state_seq[0]
                label = label_seq[0]
                initial_set = self.graph['initial'].copy()
                current_state = initial_set.pop()
            elif (t >= 1) and (len(state_seq) > t):
                # print '---observation given---'
                mdp_state = state_seq[t]
                label = label_seq[t]
                prev_state = tuple(current_state)
                error = True
                for next_state in self.successors(prev_state):
                    if((self.nodes[next_state]['mdp'] == mdp_state) and (self.nodes[next_state]['label'] == label) and (u in list(self[prev_state][next_state]['prop'].keys()))):
                        current_state = tuple(next_state)
                        error = False
                        break
                if error:
                    print(
                        'Error: The provided state and label sequences do NOT match the mdp structure!')
                    break
            else:
                # print '---random observation---'
                prev_state = tuple(current_state)
                S = []
                P = []
                if m < 2 or m == 10:                                    # in prefix or suffix, added, it is admissible for states to get in Ip again (m == 10)
                    for next_state in self.successors(prev_state):
                        prop = self[prev_state][next_state]['prop']
                        if (u in list(prop.keys())):
                            S.append(next_state)
                            P.append(prop[u][0])
                if m == 2:  # in bad states
                    # print 'in bad states'
                    Sd = best_all_plan[2][3]
                    Sf = best_all_plan[2][0]
                    Sr = best_all_plan[2][2]
                    (xf, lf, qf) = prev_state
                    postqf = self.graph['dra'].successors(qf)
                    for xt in self.graph['mdp'].successors(xf):
                        if xt != xf:
                            prop = self.graph['mdp'][xf][xt]['prop']
                            if u in list(prop.keys()):
                                prob_edge = prop[u][0]
                                label = self.graph['mdp'].nodes[xt]['label']
                                for lt in label.keys():
                                    prob_label = label[lt]
                                    dist = dict()
                                    for qt in postqf:
                                        if (xt, lt, qt) in Sf.union(Sr):
                                            dist[qt] = self.graph['dra'].check_distance_for_dra_edge(
                                                lf, qf, qt)
                                    if list(dist.keys()):
                                        qt = min(list(dist.keys()),
                                                 key=lambda q: dist[q])
                                        S.append((xt, lt, qt))
                                        P.append(prob_edge*prob_label)
                rdn = random.random()
                pc = 0
                for k, p in enumerate(P):
                    pc += p
                    if pc > rdn:
                        break
                current_state = tuple(S[k])
                mdp_state = self.nodes[current_state]['mdp']
                label = self.nodes[current_state]['label']
            # ----
            u, m = act_by_plan(self, best_all_plan, current_state)
            X.append(mdp_state)
            PX.append(current_state)
            L.append(set(label))
            U.append(u)
            M.append(m)
            t += 1
        return X, L, U, M, PX

class Sync_AMEC(DiGraph):
    def init(self):
        pass

    def synthesize_from_prod_mdps(self, amec_pi, amec_gamma, mdp_pi:Product_Dra, mdp_gamma:Product_Dra):
        # find out all amecs
        # amec:
        #   [0] amec
        #   [1] amec ^ Ip
        #   [2] action set

        #
        # 1 find all proper initial states
        stack_t = []
        visited = []

        for state_pi_t in amec_pi[0]:
            for state_gamma_t in amec_gamma[0]:
                pass

    '''
    def synthesize_from_sync_mdp(self, prod_mdp, amec, amec_prime, initial_state, observation_function=is_with_identical_observation_2):
        # amec:
        #   [0] amec
        #   [1] amec ^ Ip
        #   [2] action set

        #
        # 1 find all proper initial states
        stack_t = []
        visited = []

        # 1.1 check the available state as initial sets
        #     初始状态如何确定? 由prefix可达?
        # 1.2 based on initial sets, find all states with identical observations
        for state_i in initial_state:
            for state_j in amec_prime[0]:
                if observation_function(state_i, state_j):
                    stack_t.append((state_i, state_j))

        while stack_t.__len__():
            current_state = stack_t.pop()           # (state_in_amec, ref_state_in_amec', )
            state_in_amec = current_state[0]
            state_in_amec_prime = current_state[1]

            #
            # 3. find successive states
            # conditions:
            #   a. successor state must be in AMEC
            #   b. there exists the corresponding successor state in AMEC', s.t. the observations in a. and b. are identical
            #   c. 定量关系最后放到求解中作为约束, 不在这里求解
            #   d. the state pair must NOT be visited
            #      if visited, then check whether the edge is visited
            for state_p_in_amec in prod_mdp.successors(state_in_amec):
                for state_p_in_amec_p in prod_mdp.successors(state_in_amec_prime):          # b.1
                    if state_p_in_amec in amec[0]:                                          # a
                        if observation_function(state_p_in_amec, state_p_in_amec_p):        # b.2
                            state_tuple_p = (state_p_in_amec, state_p_in_amec_p)
                            if state_tuple_p not in visited:
                                # add edge
                                # aa. obtain probability
                                # ab. calculate differential probability


                                # add to visited
                                visited.append(state_tuple_p)
                            else:
                                # to check the edge
                                # ba. the probability is identical
                                # bb. differential probability is identical
                                pass

                # edge information
                # a. transition probabilistic of states in AMEC
                # b. absolute differential transition probabilistic between two transitions

        #
        # remeber:
        #   finally, only the AMEC with proper structure can be applied for suffix synthesis
        #   otherwise, a modified version of synthesis method must be given
        return None
    '''