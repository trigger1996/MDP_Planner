
heading_seq_list = ['N', 'E', 'S', 'W']

def sort_grids(grid, grid_p):
    global heading_seq_list

    x   = grid[0][0]
    y   = grid[0][1]
    x_p = grid_p[0][0]
    y_p = grid_p[0][1]
    try:
        heading   = grid[0][2]
        heading_p = grid_p[0][2]
        #
        heading_seq   = heading_seq_list.index(heading)
        heading_seq_p = heading_seq_list.index(heading_p)
    except IndexError:
        pass
    #
    if x > x_p:
        return 1
    elif x < x_p:
        return -1
    else:
        if y > y_p:
            return 1
        elif y < y_p:
            return -1
        else:
            try:
                if heading_seq > heading_seq_p:
                    return 1
                elif heading_seq < heading_seq_p:
                    return -1
                else:
                    return 0
            except IndexError:
                return 0

def sort_sync_grid_states(state, state_p):
    #
    grid_0     = state[0]
    grid_ref   = state[1]
    #
    grid_0_p   = state_p[0]
    grid_ref_p = state_p[1]
    sort_state_0 = sort_grids(grid_0, grid_0_p)
    if sort_state_0 != 0:
        return sort_state_0
    else:
        return sort_grids(grid_ref, grid_ref_p)

def sort_numerical_states(state, state_p):
    if int(state[0]) > int(state_p[0]):
        return 1
    elif int(state[0]) < int(state_p[0]):
        return -1
    else:
        # for product states
        if len(state) == 3:
            if int(state[2]) > int(state_p[2]):
                return 1
            elif int(state[2]) < int(state_p[2]):
                return -1
            else:
                return 0
        # else return identical
        else:
            return 0

def sort_team_numerical_states(a, b):
    state_a, label_a, num_a = a
    state_b, label_b, num_b = b

    # 比较状态元组，逐位按整数大小比较
    for sa, sb in zip(state_a, state_b):
        if int(sa) < int(sb):
            return -1
        elif int(sa) > int(sb):
            return 1
    # 若长度不同，较短的优先
    if len(state_a) < len(state_b):
        return -1
    elif len(state_a) > len(state_b):
        return 1

    # 比较标签（按字典序）
    if label_a < label_b:
        return -1
    elif label_a > label_b:
        return 1

    # 比较数值
    if num_a < num_b:
        return -1
    elif num_a > num_b:
        return 1

    return 0
