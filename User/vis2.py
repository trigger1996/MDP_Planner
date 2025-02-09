import logging
import datetime

# 生成全局日志文件名
LOG_FILE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")

# 配置日志（仅初始化一次）
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler(LOG_FILE),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
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
