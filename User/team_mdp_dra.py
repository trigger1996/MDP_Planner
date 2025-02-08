from MDP_TG.dra import Product_Dra
from User.dra2 import product_mdp2
from MDP_TG.mdp import Motion_MDP
from networkx import DiGraph
import itertools


def find_individual_label_for_team_state(state, mdp_list):
    label  = []
    for i in range(0, mdp_list.__len__()):
        mdp_t = mdp_list[i]
        state_t = state[i]
        state_attr_t = mdp_t.nodes()[state_t]
        label_t = state_attr_t['label']
        label.append(label_t)
    return label

def generate_team_label(state, mdp_list):
    label_list = find_individual_label_for_team_state(state, mdp_list)
    label_tuple = []
    prob_list = []
    for label_i in label_list:
        label_i_t = list(label_i.keys())[0]         # assume there is only one key for each key
        prob_val  = label_i[label_i_t]              #       i.e., only one label for each state
        label_tuple.append(label_i_t)
        prob_list.append(prob_val)
    label_tuple = tuple(label_tuple)
    return label_tuple, prob_list


class Team_MDP(Motion_MDP):
    def __init__(self, mdp_list):
        init_node = []
        init_label = []
        for mdp_t in mdp_list:
            #
            init_node_i = mdp_t.graph['init_state']
            init_node.append(init_node_i)
            #
            init_label_i = mdp_t.graph['init_label']
            init_label.append(init_label_i)
        init_node  = tuple(init_node)
        init_label = tuple(init_label)

        DiGraph.__init__(self, name='motion_mdp',
                         init_state=init_node, init_label=init_label)

        self.dfs_initialization(init_node, mdp_list)


    def dfs_initialization(self, init_node, mdp_list):

        stack = [ init_node ]
        visited = []

        while stack.__len__():
            #
            current_state = stack.pop()
            #
            if current_state in visited:
                continue
            visited.append(current_state)
            #
            label_p, prob_t = generate_team_label(current_state, mdp_list)
            self.add_node(current_state,label={label_p : prob_t}, act=set())

            current_state_i_list = []
            for state_i in current_state:
                current_state_i_list.append(state_i)

            suc_state_list = []
            for i in range(0, mdp_list.__len__()):
                suc_state_list_t = list(mdp_list[i].successors(current_state_i_list[i]))
                suc_state_list.append(suc_state_list_t)
            #
            suc_state_list_p = list(itertools.product(*suc_state_list))         # *表示解包

            for state_p in suc_state_list_p:
                event_p_list = []
                prop_p_list = []
                cost_p_list = []
                for i in range(0, mdp_list.__len__()):
                    state_i      = current_state[i]
                    state_i_next = state_p[i]
                    #
                    event_i = list(mdp_list[i].edges[state_i, state_i_next]['prop'].keys())[0]
                    prop_i = mdp_list[i].edges[state_i, state_i_next]['prop'][event_i][0]
                    cost_i = mdp_list[i].edges[state_i, state_i_next]['prop'][event_i][1]
                    #
                    event_p_list.append(event_i)
                    prop_p_list.append(prop_i)
                    cost_p_list.append(cost_i)

                #
                event_p_list = tuple(event_p_list)
                # CRITICAL
                # calculate cost and probability
                # considering the probability of each system is independent, so the probabilities can be directly multiplied
                # as for the cost, we use the MAXIMAL value instead of the individual value
                # such that all system will move synchronously, and the faster ones will wait for the slower ones
                prop_p_final = 1.
                for prop_p in prop_p_list:
                    prop_p_final *= prop_p
                #
                cost_p_final = max(cost_p_list)

                self.add_edge(current_state, state_p, prop={event_p_list : [prop_p_final, cost_p_final, prop_p_list]})      # joint probability, cost, separated probability
                stack.append(state_p)

        U = []
        for f_node in self.nodes():
            Us = set()
            for t_node in self.successors(f_node):
                prop = self[f_node][t_node]['prop']
                Us.update(set(prop.keys()))
            if Us:
                self.nodes[f_node]['act'] = Us.copy()
                U = U + [U_t for U_t in Us]
                U = list(set(U))
            else:
                print('Isolated state')
        #
        self.graph['U'] = set(U)

class Team_Product_Dra(product_mdp2):                       # modified, Team_Product_Dra(Product_Dra)
    def __init__(self, mdp, dra):
        product_mdp2.__init__(self, mdp=mdp, dra=dra)       # Product_Dra.__init__(......)

        self.name = 'Team_Product_Dra'

    def build_full(self):
        # TODO
        # if there exist mutiple initial states
        mdp_initial_node      = self.graph['mdp'].graph['init_state']
        mdp_initial_label     = self.graph['mdp'].graph['init_label']
        dra_initial_node_list = list(self.graph['dra'].graph['initial'])

        stack = [ ]
        for dra_initial_node_t in dra_initial_node_list:
            f_prod_node = self.composition(mdp_initial_node, mdp_initial_label, dra_initial_node_t)
            stack.append(f_prod_node)
        visited = []
        while stack.__len__():
            #
            # USE depth-first-search method
            f_prod_node = stack.pop()
            if f_prod_node in visited:
                continue
            visited.append(f_prod_node)
            visited = list(set(visited))

            f_mdp_node  = f_prod_node[0]
            f_mdp_label = f_prod_node[1]
            f_dra_node  = f_prod_node[2]
            for t_mdp_node in self.graph['mdp'].successors(f_mdp_node):
                mdp_edge = self.graph['mdp'][f_mdp_node][t_mdp_node]
                for t_mdp_label, t_label_prob in self.graph['mdp'].nodes[t_mdp_node]['label'].items():
                    for t_dra_node in self.graph['dra'].successors(f_dra_node):
                        #
                        # establish successor states
                        t_prod_node = self.composition(t_mdp_node, t_mdp_label, t_dra_node)
                        #
                        # check whether successor states satisfies DRA conditions
                        truth = self.check_label_for_dra_edge(f_mdp_label, f_dra_node, t_dra_node)
                        if truth:                                                                                       # DRA condition, not accepting condition, so if DRA condition is satisfied, then the system can step forward
                            #
                            # Add successor product dra states to to-visit list
                            stack.append(t_prod_node)
                            #
                            prob_cost = dict()
                            #
                            # calculate
                            for u, attri in mdp_edge['prop'].items():
                                #neg_joint_expectation = 1.
                                joint_expectation = 1.
                                for i in range(0, t_label_prob.__len__()):                                              # t_label_prob is a list for transition probability for each agent
                                    if True:                                                                            # dra condition not accepting condition
                                        # neg_joint_expectation *= (1. - t_label_prob[i]) * (1. - attri[2][i])          # none of these aps are satisfied
                                        joint_expectation *= t_label_prob[i] * attri[2][i]
                                #joint_expectation = 1. - neg_joint_expectation
                                joint_expectation = joint_expectation                                                   # to distinguish
                                if joint_expectation != 0:                                                              # TODO, here, we pick the possibility that ALL APs are satisfied
                                    # prob_cost[u] = (t_label_prob * attri[0], attri[1])                                # for single-agent system
                                    prob_cost[u] = (joint_expectation, attri[1])                                        # attr[0] labeling probability / attr[1] cost / attr[2] final
                            if list(prob_cost.keys()):
                                self.add_edge(f_prod_node, t_prod_node, prop=prob_cost)
        #
        # once dfs method completed
        self.build_acc()
        print("-------Prod DRA Constructed-------")
        print("%s states, %s edges and %s accepting pairs" % (
            str(len(self.nodes())), str(len(self.edges())), str(len(self.graph['accept']))))

    def check_label_for_dra_edge(self, label, f_dra_node, t_dra_node):
        # ----check if a label satisfies the guards on one dra edge----
        guard_string_list = self.graph['dra'][f_dra_node][t_dra_node]['guard_string']
        guard_int_list = []
        for st in guard_string_list:
            int_st = []
            for l in st:
                int_st.append(int(l))
            guard_int_list.append(int_st)
        for guard_list in guard_int_list:
            valid = True
            for k, ap in enumerate(self.graph['dra'].graph['symbols']):
                if (guard_list[k] == 1) and (not self.ap_in_label_list(ap, label)):     # (ap not in label):
                    valid = False
                if (guard_list[k] == 0) and self.ap_in_label_list(ap, label):           # (ap in label):
                    valid = False
            if valid:
                # That is,
                # ap in label list and guard_list[k] == 1, or
                # ap NOT in label list and guard_list[k] == 0
                return True
        return False

    def ap_in_label_list(self, ap, label):
        for label_i in label:
            if ap in label_i:
                return True
        return False