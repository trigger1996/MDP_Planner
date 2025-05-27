import os
import logging
import datetime
import random

from collections import Counter
from colorama import Fore, Back, Style, init
from typing import Union, List
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


def print_c(
        data,
        color: Union[int, str] = "green",
        style: Union[str, List[str]] = None,
        bg_color: Union[int, str] = None,
        log_file: str = LOG_FILE,
        **kwargs
):
    """
    增强版颜色样式打印输出功能，支持多种文本样式和背景色

    参数:
        data: 要打印的内容
        color: 颜色，可以是数字(30-37)或字符串名称(如'red','green')
        style: 文本样式，支持: 'bold','italic','underline','blink','reverse'
        bg_color: 背景色，可以是数字(40-47)或字符串名称(如'bg_red','bg_green')
        log_file: 日志文件路径，如果为None则不记录日志
        kwargs: 其他日志配置参数

    颜色代码:
        前景色: 黑色(30), 红色(31), 绿色(32), 黄色(33),
               蓝色(34), 紫色(35), 青色(36), 白色(37)
        背景色: 黑色(40), 红色(41), 绿色(42), 黄色(43),
               蓝色(44), 紫色(45), 青色(46), 白色(47)
    """
    # 颜色和样式映射表
    color_map = {
        'black': 30, 'red': 31, 'green': 32, 'yellow': 33,
        'blue': 34, 'magenta': 35, 'cyan': 36, 'white': 37,
        'bg_black': 40, 'bg_red': 41, 'bg_green': 42, 'bg_yellow': 43,
        'bg_blue': 44, 'bg_magenta': 45, 'bg_cyan': 46, 'bg_white': 47
    }

    style_map = {
        'bold': 1, 'italic': 3, 'underline': 4,
        'blink': 5, 'reverse': 7
    }

    # 处理颜色参数
    if isinstance(color, str):
        color_code = color_map.get(color.lower(), 32)  # 默认绿色
    else:
        color_code = color if 30 <= color <= 37 else 32

    # 处理样式参数
    style_codes = []
    if style:
        if isinstance(style, str):
            style = [style]
        for s in style:
            if s.lower() in style_map:
                style_codes.append(str(style_map[s.lower()]))

    # 处理背景色参数
    bg_code = ""
    if bg_color:
        if isinstance(bg_color, str):
            bg_code = str(color_map.get(f"bg_{bg_color.lower()}", 40))
        elif 40 <= bg_color <= 47:
            bg_code = str(bg_color)

    # 构建ANSI转义序列
    codes = style_codes + [str(color_code)]
    if bg_code:
        codes.append(bg_code)
    ansi_start = "\033[" + ";".join(codes) + "m"
    ansi_end = "\033[0m"

    # 打印带样式的文本
    print(f"{ansi_start}{data}{ansi_end}")

    # 记录日志
    if log_file:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            **kwargs
        )
        data_str = str(data)
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
    """彩色打印序列中的 Counter 项"""
    for item in sequence:
        if isinstance(item, list):
            print('[', end='')
            for subitem in item:
                color = get_color_for_item(subitem)
                print(f"{color}'{subitem}'{Fore.RESET}, ", end='')
            print(']', end=' ')
        elif isinstance(item, Counter):
            print('{', end='')
            for k, v in item.items():
                color = get_color_for_item(k)
                print(f"{color}'{k}':{v}{Fore.RESET}, ", end='')
            print('}', end=' ')
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
    for i in range(0, X_U.__len__() - 1, 2):
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
