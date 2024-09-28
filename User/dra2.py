from math import fabs, sqrt
from networkx import DiGraph
from MDP_TG.mdp import find_MECs, find_SCCs
from MDP_TG.dra import Product_Dra

def observation_func_1(state, observation='X'):
    if observation == 'X':
        return state[0][0]
    if observation == 'y':
        return state[0][1]     # x and y are NOT inverted here, they are only inverted in plotting

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


def find_initial_state(y_in_sf, state_set_gamma, is_observed_identical=is_with_identical_observation_1):
    state_list = []
    for state_pi_t in y_in_sf.keys():
        if y_in_sf[state_pi_t] == 0:
            continue
        for state_gamma_t in state_set_gamma:
            #
            if is_observed_identical(state_pi_t, state_gamma_t):
                if is_ap_identical(state_pi_t, state_gamma_t):
                    state_list.append((state_pi_t, state_gamma_t, ))
    return state_list

class product_mdp2(Product_Dra):
    def __init__(self, mdp, dra):
        Product_Dra.__init__(self, mdp, dra)
        self.compute_S_f()
        #
        self.sync_amec_set = list()

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

    def re_synthesize_sync_amec(self, y_in_sf, MEC_pi, MEC_gamma, product_mdp_gamma:Product_Dra, is_observed_identical=is_with_identical_observation_3, is_re_compute_Sf=True):
        # amec:
        #   [0] amec
        #   [1] amec ^ Ip
        #   [2] action set
        #
        mec_state_set_pi    = MEC_pi[0]
        mec_state_set_gamma = MEC_gamma[0]

        stack_t = find_initial_state(y_in_sf, list(MEC_gamma[0]))
        stack_t = list(set(stack_t))
        visited = []

        sync_mec_t = DiGraph()
        for state_t in stack_t:
            sync_mec_t.add_node(state_t)
        while stack_t.__len__():
            current_state = stack_t.pop()
            visited.append(current_state)
            #
            next_state_list_pi    = list(self.out_edges(current_state[0]))
            next_state_list_gamma = list(product_mdp_gamma.out_edges(current_state[1]))
            #
            for edge_t_pi in next_state_list_pi:
                for edge_t_gamma in next_state_list_gamma:
                    #
                    next_state_pi    = edge_t_pi[1]
                    next_state_gamma = edge_t_gamma[1]
                    next_sync_state = (next_state_pi, next_state_gamma)
                    #
                    # 要求这个状态没有被考虑过
                    if next_sync_state in visited:
                        continue
                    #
                    is_next_state_pi_in_amec    = next_state_pi    in mec_state_set_pi
                    is_next_state_gamma_in_amec = next_state_gamma in mec_state_set_gamma
                    if not is_next_state_pi_in_amec or not is_next_state_gamma_in_amec:
                        continue
                    #
                    if is_observed_identical(next_state_pi, next_state_gamma):
                        if is_ap_identical(next_state_pi, next_state_gamma):
                            #
                            # TODO
                            # 状态转移概率和cost
                            sync_mec_t.add_edge(current_state, next_sync_state)
                            #
                            if next_sync_state not in stack_t:
                                stack_t.append(next_sync_state)
        #
        if sync_mec_t.edges().__len__():
            #
            # TODO
            # 检查连接性
            for state_sync_t in sync_mec_t.nodes():
                pass
            #
            self.sync_amec_set.append(sync_mec_t)


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
