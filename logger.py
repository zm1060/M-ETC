import logging
import os

# 设置日志记录
def setup_logging(log_file='experiment.log'):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(
        filename=os.path.join('logs', log_file),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# 记录实验信息
def log_experiment_info(params, accuracy, loss):
    logging.info(f'Parameters: {params}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}')
