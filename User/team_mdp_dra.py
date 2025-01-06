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

                self.add_edge(current_state, state_p, prop={event_p_list : [prop_p_final, cost_p_final]})
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