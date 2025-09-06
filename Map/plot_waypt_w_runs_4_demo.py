import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re
from collections import deque

# === 载入地图数据 ===
# 0426 Multi example
yaml_file = "./yaml/20250426_map_w_edges.yaml"
with open(yaml_file, "r") as f:
    data = yaml.load(f, Loader=yaml.UnsafeLoader)

waypoints = data["waypoint"]
edges = data["edges"]

# === 原始运行序列（状态对 + 动作对） ===
raw_sequence = '''
    '('0', '5')' '(('b',), ('b',))' '('2', '6')' '(('a',), ('a',))' '('3', '4')' '(('b',), ('b',))' '('0', '2')' '(('b',), ('a',))' '('2', '3')' '(('a',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('0', '2')' '(('b',), ('a',))' '('2', '3')' '(('a',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('0', '2')' '(('b',), ('a',))' '('2', '3')' '(('a',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('4', '3')' '(('a',), ('b',))' '('5', '4')' '(('b',), ('a',))' '('6', '3')' '(('b',), ('b',))' '('6', '0')' '(('a',), ('a',))' '('4', '1')' '(('b',), ('a',))' '('2', '0')' '(('a',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')' '(('a',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('0', '3')'
'''
# '('0', '5')' '(('b',), ('b',))' '('2', '6')' '(('a',), ('a',))' '('3', '4')' '(('b',), ('b',))' '('0', '2')' '(('b',), ('a',))' '('2', '3')' '(('a',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('0', '2')' '(('b',), ('a',))' '('2', '3')' '(('a',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('0', '2')' '(('b',), ('a',))' '('2', '3')' '(('a',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('4', '3')' '(('a',), ('b',))' '('5', '4')' '(('b',), ('a',))' '('6', '3')' '(('b',), ('b',))' '('6', '0')' '(('a',), ('a',))' '('4', '1')' '(('b',), ('a',))' '('2', '0')' '(('a',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')' '(('a',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')' '(('a',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('0', '3')' '(('b',), ('b',))' '('3', '4')' '(('b',), ('a',))' '('0', '5')' '(('a',), ('a',))' '('1', '4')' '(('a',), ('a',))' '('0', '5')' '(('b',), ('b',))' '('0', '6')' '(('a',), ('b',))' '('1', '6')' '(('a',), ('b',))' '('0', '6')' '(('a',), ('b',))' '('1', '6')' '(('a',), ('b',))' '('0', '6')' '(('b',), ('a',))' '('3', '4')' '(('b',), ('a',))' '('0', '5')' '(('a',), ('b',))' '('1', '6')' '(('a',), ('b',))' '('0', '6')' '(('b',), ('a',))' '('2', '4')' '(('a',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')' '(('a',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('0', '3')' '(('b',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('0', '3')' '(('b',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')'
# '('0', '5')' '(('b',), ('b',))' '('2', '6')' '(('a',), ('a',))' '('3', '4')' '(('b',), ('b',))' '('0', '2')' '(('b',), ('a',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')' '(('a',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')' '(('a',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('0', '3')'
##
#
# good points, vise vesa
# (2, 3), (2, 6)
# bad points
# (1, *))
#
# for comparasion
# 可以看到non-opaque runs (如下)是会走（1，0）之类的点的
# '('0', '5')' '(('a',), ('a',))' '('1', '4')' '(('a',), ('a',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')' '(('a',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('0', '3')' '(('a',), ('b',))' '('1', '0')' '(('a',), ('a',))' '('0', '1')' '(('b',), ('a',))' '('2', '0')' '(('a',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')' '(('a',), ('a',))' '('3', '1')' '(('b',), ('a',))' '('4', '0')' '(('a',), ('a',))' '('5', '1')' '(('b',), ('a',))' '('6', '0')' '(('a',), ('b',))' '('4', '3')' '(('a',), ('b',))' '('5', '4')' '(('a',), ('a',))' '('4', '5')' '(('b',), ('b',))' '('2', '6')' '(('a',), ('a',))' '('3', '4')' '(('b',), ('a',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')' '(('a',), ('a',))' '('3', '1')' '(('b',), ('a',))' '('4', '0')' '(('b',), ('b',))' '('2', '3')' '(('a',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('0', '2')' '(('a',), ('a',))' '('1', '3')' '(('a',), ('b',))' '('0', '4')' '(('b',), ('a',))' '('3', '5')' '(('b',), ('a',))' '('0', '4')' '(('b',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('0', '3')' '(('a',), ('b',))' '('1', '0')' '(('a',), ('b',))' '('0', '2')' '(('b',), ('a',))' '('2', '3')' '(('a',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('0', '3')' '(('a',), ('b',))' '('1', '0')' '(('a',), ('b',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')' '(('a',), ('b',))' '('3', '2')' '(('b',), ('a',))' '('4', '3')' '(('a',), ('b',))' '('5', '0')' '(('b',), ('b',))' '('6', '3')' '(('a',), ('b',))' '('4', '0')' '(('b',), ('b',))' '('2', '0')' '(('a',), ('b',))' '('3', '0')' '(('b',), ('b',))' '('0', '3')' '(('b',), ('b',))' '('2', '0')'


#
# 0506 multi agent


# === 提取多机器人状态序列 ===
state_tuples = re.findall(r"\((?:'(\d+)'(?:, )?)+\)", raw_sequence)
# 上面只能提取第一个，改通用：
state_matches = re.findall(r"\((.*?)\)", raw_sequence)
multiagent_states = []
for match in state_matches:
    parts = [p.strip(" '") for p in match.split(",") if p.strip()]
    if all(p.isdigit() for p in parts):
        multiagent_states.append(tuple(parts))

num_agents = len(multiagent_states[0])
print(f"共提取到 {len(multiagent_states)} 个多智能体状态，每个状态 {num_agents} 个机器人")

# === 准备底图 ===
fig, ax = plt.subplots(figsize=(8, 6))

# 画节点
node_positions = {}
for node_id, node_info in waypoints.items():
    x_enu, y_enu, z, yaw = node_info["pos"]
    ap = node_info["ap"][0]
    Xned, Yned = y_enu, x_enu
    node_positions[node_id] = (Xned, Yned)
    ax.scatter(Xned, Yned, c="skyblue", edgecolors="black", s=200, zorder=3)
    ax.text(Xned, Yned + 0.1, f"{node_id}\n{ap}", ha="center", va="bottom", fontsize=8)

# 画边 (静态背景)
for src, dst, attr in edges:
    x1, y1, _, _ = waypoints[src]["pos"]
    x2, y2, _, _ = waypoints[dst]["pos"]
    X1, Y1 = y1, x1
    X2, Y2 = y2, x2
    ax.arrow(X1, Y1, X2 - X1, Y2 - Y1,
             length_includes_head=True,
             head_width=0.1, head_length=0.15,
             fc="gray", ec="gray", alpha=0.2, zorder=1)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
ax.set_title(data["name"])
ax.grid(True, linestyle="--", alpha=0.5)

# === 动画对象 ===
colors = ["red", "green", "orange", "purple", "blue", "pink"]
agents = [ax.plot([], [], "o", color=colors[i % len(colors)],
                  markersize=12, zorder=4, label=f"Robot {i+1}")[0]
          for i in range(num_agents)]

# 每个机器人维护尾迹线段（deque）
max_tail = 5  # 尾迹长度
tails = [deque(maxlen=max_tail) for _ in range(num_agents)]
tail_lines = [ [] for _ in range(num_agents)]  # 动态保存 Line2D

def init():
    for agent in agents:
        agent.set_data([], [])
    return agents

def update(frame):
    global tail_lines
    # 清除旧的尾迹线
    for line in sum(tail_lines, []):
        line.remove()
    tail_lines = [[] for _ in range(num_agents)]

    state = multiagent_states[frame]
    if frame > 0:
        prev_state = multiagent_states[frame-1]
        # 更新尾迹队列
        for i in range(num_agents):
            tails[i].append((prev_state[i], state[i]))

    for i, pos in enumerate(state):
        x, y = node_positions[pos]
        agents[i].set_data(x, y)

        # 画尾迹
        for j, (s, t) in enumerate(tails[i]):
            x1, y1 = node_positions[s]
            x2, y2 = node_positions[t]
            alpha = (j+1)/len(tails[i]) * 0.6  # 越新越亮
            line, = ax.plot([x1, x2], [y1, y2],
                            color=colors[i % len(colors)],
                            linewidth=2, alpha=alpha, zorder=2)
            tail_lines[i].append(line)

    return agents + sum(tail_lines, [])

ani = FuncAnimation(fig, update, frames=len(multiagent_states),
                    init_func=init, interval=800, blit=True, repeat=True)

ax.legend()
plt.show()
