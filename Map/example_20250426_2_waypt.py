#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
from collections import OrderedDict

# 替换为你实际的模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MDP_Planner'))
from example_20250426_team_mdp import build_individual_mdp  # <- 修改为实际模块路径


def assign_fake_positions(node_ids, spacing=2.0):
    # """为每个节点沿X轴分配虚拟坐标，仅用于可视化"""
    # pos_map = {}
    # for i, node_id in enumerate(sorted(node_ids, key=int)):
    #     # TODO
    #     pos_map[node_id] = (i * spacing, 0.0)  # 仅沿X轴分布
    pose_multiplier = 1.5
    raw_pos_map = {
        '0' : (0.,    0.,   1.2, 90.),
        '1' : (2.,    1.5,  1.2, 90.),
        '2' : (2.5,   0.,   1.2, 90.),
        '3' : (1.75, -1.5,  1.2, 90.),
        '4' : (3.75,  0.,   1.2, 90.),
        '5' : (5.75,  0.,   1.2, 90.),
        '6' : (3.75,  1.5,  1.2, 90.),
    }

    # 对 x 和 y 乘以 pose_multiplier，z 和 yaw 保持不变
    scaled_pos_map = {
        node_id: (x * pose_multiplier, y * pose_multiplier, z, yaw)
        for node_id, (x, y, z, yaw) in raw_pos_map.items()
    }

    return scaled_pos_map

def convert_nodes_to_waypoints(robot_nodes_w_aps, node_positions, z=1.2, yaw=90.0, transition_time=0):
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


def generate_individual_mdp_yaml(output_path='./yaml/20250426_map_w_edges.yaml'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 调用你的构建函数
    robot_nodes_w_aps, robot_edges, _, initial_node, _ = build_individual_mdp()

    # 自动给每个节点分配坐标（可用于展示）
    node_positions = assign_fake_positions(robot_nodes_w_aps.keys())

    yaml_data = OrderedDict()
    yaml_data['name'] = '20250426_map_w_edges'
    yaml_data['initial_node'] = initial_node
    yaml_data['waypoint'] = convert_nodes_to_waypoints(robot_nodes_w_aps, node_positions)
    yaml_data['edges'] = convert_edges_to_yaml_format(robot_edges)

    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"✅ YAML saved to {output_path}")


if __name__ == '__main__':
    generate_individual_mdp_yaml()
