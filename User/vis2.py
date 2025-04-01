import os
import logging
import datetime
import random
from colorama import Fore, Back, Style, init
import re

# 创建日志文件夹（如果不存在）
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 生成全局日志文件名
#LOG_FILE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
LOG_FILE = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))


# 配置日志（仅初始化一次）
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(message)s',     # 日志格式
    handlers=[
        logging.FileHandler(LOG_FILE),      # 输出到文件
        #logging.StreamHandler()            # 输出到控制台
    ]
)


def print_c(data, color=32):
    """
    颜色样式打印输出功能，并支持将数据写入日志文件。

    :param data: 打印内容
    :param color: 指定颜色, 默认为绿色(32)
    :return:
    """
    # 颜色打印
    if isinstance(color, int):
        color = str(color)
    print(f"\033[1;{color}m{data}\033[0m")

    # 如果数据是列表或字典，转换为字符串
    data_str = str(data)

    # 记录到日志文件
    logging.info(data_str)


# 初始化 colorama
init(autoreset=True)

def get_color_for_item(item):
    """为不同的字母/单词分配不同的颜色"""
    if isinstance(item, tuple):
        item = item[0]  # 处理 ('a',) 这样的情况
    
    # 字母到颜色的映射
    color_map = {
        'a': Fore.RED,
        'b': Fore.GREEN,
        'c': Fore.BLUE,
        'd': Fore.YELLOW,
        'u': Fore.CYAN,
        'v': Fore.MAGENTA,
        'w': Fore.WHITE + Back.BLACK,
        'x': Fore.LIGHTRED_EX,
        'recharge': Fore.GREEN,
        'drop': Fore.RED,
        'gather': Fore.BLUE,
        # 数字可以保持默认颜色或另外指定
    }
    
    # 如果是数字，保持默认颜色
    if isinstance(item, str) and item.isdigit():
        return Fore.RESET
    
    return color_map.get(item, Fore.RESET)

def print_colored_sequence(sequence):
    """彩色打印序列"""
    for item in sequence:
        if isinstance(item, list):
            # 处理嵌套列表
            print('[', end='')
            for subitem in item:
                color = get_color_for_item(subitem)
                print(f"{color}'{subitem}'{Fore.RESET}, ", end='')
            print(']', end=' ')
        else:
            color = get_color_for_item(item)
            print(f"{color}'{item}'{Fore.RESET}", end=' ')
    print()

def get_display_length(s):
    """计算字符串的显示长度（忽略颜色代码）"""
    return len(re.sub(r'\033\[[0-9;]*m', '', str(s)))

def pad_to_width(s, width):
    """将字符串填充到指定宽度（考虑颜色代码）"""
    actual_width = get_display_length(s)
    return s + ' ' * (width - actual_width)

def print_highlighted_sequences(X_U, Y, X_INV, AP_INV, marker1='ap_pi', marker2='ap_gamma', attr='Opaque'):
    """
    双标记高亮打印，同时标记ap_pi和ap_gamma

    参数:
        X_U, Y, X_INV, AP_INV: 四个对应的序列
        marker1: 第一个标记词 (默认'ap_pi')
        marker2: 第二个标记词 (默认'ap_gamma')
    """
    # 0. 处理X_U序列
    X_U_p = []
    for i in range(0, X_U.__len__(), 2):
        X_U_p.append((X_U[i], X_U[i + 1]))

    # 1. 在AP_INV中找到所有标记位置
    marker1_positions = [i for i, item in enumerate(AP_INV)
                        if isinstance(item, list) and marker1 in item]
    marker2_positions = [i for i, item in enumerate(AP_INV)
                        if isinstance(item, list) and marker2 in item]

    # 合并所有标记位置并排序
    all_markers = sorted([(pos, 1) for pos in marker1_positions] +
                         [(pos, 2) for pos in marker2_positions])

    if len(all_markers) < 2:
        print("Not enough markers found")
        return

    # 2. 设置颜色方案
    marker1_color = Fore.YELLOW + Back.BLUE  # ap_pi的颜色
    marker2_color = Fore.WHITE + Back.RED    # ap_gamma的颜色
    interval_color = Fore.CYAN               # 区间颜色

    # 3. 设置列宽
    col_widths = {
        'index': 6,
        'X_U': 40,
        'Y': 30,
        'X_INV': 40,
        'AP_INV': 40
    }

    # 4. 打印表头
    header = (f"{pad_to_width('Index', col_widths['index'])}"
              f"{pad_to_width('X_U', col_widths['X_U'])}"
              f"{pad_to_width('Y', col_widths['Y'])}"
              f"{pad_to_width('X_INV', col_widths['X_INV'])}"
              f"{pad_to_width('AP_INV', col_widths['AP_INV'])}")
    print(attr)
    print(header)
    print("-" * sum(col_widths.values()))

    # 5. 打印对齐的内容
    current_marker_idx = 0
    in_highlight = False

    for pos in range(max(len(X_U_p), len(Y), len(X_INV), len(AP_INV))):
        # 检查是否到达标记位置
        if current_marker_idx < len(all_markers) and pos == all_markers[current_marker_idx][0]:
            marker_type = all_markers[current_marker_idx][1]
            in_highlight = not in_highlight
            if not in_highlight:
                current_marker_idx += 1

        # 收集各列内容
        cols = {
            'index': str(pos),
            'X_U': X_U_p[pos] if pos < len(X_U_p) else "",
            'Y': Y[pos] if pos < len(Y) else "",
            'X_INV': X_INV[pos] if pos < len(X_INV) else "",
            'AP_INV': AP_INV[pos] if pos < len(AP_INV) else ""
        }

        # 格式化每列内容
        formatted_cols = {}
        for name, item in cols.items():
            if name == 'index':
                formatted_cols[name] = pad_to_width(item, col_widths[name])
                continue

            item_str = str(item)

            # 判断标记类型
            if isinstance(item, list):
                if marker1 in item:
                    display_str = f"{marker1_color}{item_str}{Style.RESET_ALL}"
                elif marker2 in item:
                    display_str = f"{marker2_color}{item_str}{Style.RESET_ALL}"
                elif in_highlight:
                    display_str = f"{interval_color}{item_str}{Style.RESET_ALL}"
                else:
                    display_str = item_str
            else:
                if in_highlight:
                    display_str = f"{interval_color}{item_str}{Style.RESET_ALL}"
                else:
                    display_str = item_str

            formatted_cols[name] = pad_to_width(display_str, col_widths[name])

        # 打印行
        print(f"{formatted_cols['index']}"
              f"{formatted_cols['X_U']}"
              f"{formatted_cols['Y']}"
              f"{formatted_cols['X_INV']}"
              f"{formatted_cols['AP_INV']}")

    print()
