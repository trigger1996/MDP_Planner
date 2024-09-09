from math import fabs, sqrt
from networkx import DiGraph
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

def is_with_identical_observation_3(state, state_prime, dist_threshold=1.):
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

class Sync_Product_Dra(DiGraph):
    def init(self):
        pass

    def synthesize_from_sync_mdp(self, prod_mdp, amec, amec_prime, initial_state, observation_function=is_with_identical_observation_2):

        #
        # 1 find all proper initial states
        stack_t = []
        visited_in_stack = []

        # 1.1 check the available state as initial sets
        #     初始状态如何确定? 由prefix可达?
        # 1.2 based on initial sets, find all states with identical observations
        for state_i in initial_state:
            for state_j in amec_prime[0]:
                if observation_function(state_i, state_j):
                    stack_t.append((state_i, state_j))

        while stack_t.__len__():
            current_state = stack_t.pop()           # (state_in_amec, ref_state_in_amec', )
            stack_dfs  = [ current_state ]
            visited_dfs = []

            #
            # 2. make depth_first_search from current_state
            while stack_dfs.__len__():
                dfs_state = stack_dfs.pop()
                state_in_amec = dfs_state[0]
                state_in_amec_prime = dfs_state[1]

                #
                # 3. find successive states
                # conditions:
                #   a.
                #   b.
                #   c.
                for state_p_in_amec in prod_mdp.successors(state_in_amec):
                    pass

        return None
