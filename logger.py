import logging
import os

# 创建logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 创建handler，用于写入日志文件
logfile = '~/logs/insightface/recognition.log'
logfile = os.path.expanduser(logfile)
logdir = os.path.dirname(logfile)
if not os.path.exists(logdir):
    os.makedirs(logdir)
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)

# 创建handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# 定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 将logger添加到handler里面
# logger.addHandler(fh)
logger.addHandler(ch)
