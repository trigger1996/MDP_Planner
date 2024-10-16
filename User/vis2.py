
def print_c(data, color=32):
    """
    https://blog.csdn.net/XianZhe_/article/details/113075983
    颜色样式打印输出功能
    :param data: 打印内容
    :param color: 指定颜色, 默认为绿色(32)
    :return:
    """
    if isinstance(color, int):
        color = str(color)
    print(f"\033[1;{color}m{data}\033[0m")