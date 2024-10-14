
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
