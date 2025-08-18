#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
from collections import OrderedDict

# 替换为你实际的模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MDP_Planner'))
from example_20250426_team_mdp import build_individual_mdp  # <- 修改为实际模块路径


def assign_fake_positions(node_ids, scale=1.0, x_offset=0.0, y_offset=0.0):
    """给每个节点分配虚拟坐标，可加比例系数和平移偏移"""
    raw_pos_map = {
        '0': (0.,    0.,   1.2, 90.),
        '1': (2.,    1.5,  1.2, 90.),
        '2': (2.5,   0.,   1.2, 90.),
        '3': (1.75, -1.5,  1.2, 90.),
        '4': (3.75,  0.,   1.2, 90.),
        '5': (5.75,  0.,   1.2, 90.),
        '6': (3.75,  1.5,  1.2, 90.),
    }

    # 对 x 和 y 做缩放+偏移，z 和 yaw 保持不变
    transformed_pos_map = {
        node_id: (x * scale + x_offset, y * scale + y_offset, z, yaw)
        for node_id, (x, y, z, yaw) in raw_pos_map.items()
    }

    return transformed_pos_map


def convert_nodes_to_waypoints(robot_nodes_w_aps, node_positions, transition_time=0):
    waypoints = OrderedDict()
    for node_id, (x, y, z, yaw) in node_positions.items():
        waypoints[str(node_id)] = {
            "pos": [x, y, z, yaw],
            "ap": [str(set(ap_t)) for ap_t in list(robot_nodes_w_aps[node_id].keys())],
            "transition": transition_time
        }
    return waypoints


def convert_edges_to_yaml_format(robot_edges):
    edges = []
    for (u, act, v), (prob, cost) in robot_edges.items():
        edges.append([u, v, {'control': act, 'weight': round(cost)}])
    return edges


def generate_individual_mdp_yaml(output_path='./yaml/20250426_map_w_edges.yaml',
                                 scale=0.85, x_offset=-4.0, y_offset=-4.0):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 调用你的构建函数
    robot_nodes_w_aps, robot_edges, _, initial_node, _ = build_individual_mdp()

    # 给节点分配坐标（可缩放 & 偏移）
    node_positions = assign_fake_positions(robot_nodes_w_aps.keys(),
                                           scale=scale,
                                           x_offset=x_offset,
                                           y_offset=y_offset)

    yaml_data = OrderedDict()
    yaml_data['name'] = '20250426_map_w_edges'
    yaml_data['initial_node'] = initial_node
    yaml_data['waypoint'] = convert_nodes_to_waypoints(robot_nodes_w_aps, node_positions)
    yaml_data['edges'] = convert_edges_to_yaml_format(robot_edges)

    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"✅ YAML saved to {output_path}")


if __name__ == '__main__':
    # 示例：整体放大 1.5 倍，并平移 (x=2.0, y=1.0)
    generate_individual_mdp_yaml(output_path='./yaml/20250426_map_w_edges.yaml')
