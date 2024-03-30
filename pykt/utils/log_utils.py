import logging
import uuid

# 配置 logging
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
    handlers=[logging.StreamHandler()],
)  # 将日志输出到控制台
