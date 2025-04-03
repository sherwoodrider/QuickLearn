import datetime
import os

from src.model.load_model import TinyLlamaLoader
from src.resource_monitor.monitor import ResourceMonitor
from src.utility.gradio_lunch import GradioClass


def get_save_log_path(test_path):
    now = datetime.datetime.now()
    # 格式化时间为文件名格式
    str_now = now.strftime('%Y_%m_%d_%H_%M_%S')
    log_folder_name = "quick_learn_" + str_now
    save_log_folder = os.path.join(test_path, "logs")
    test_log_folder = os.path.join(save_log_folder, log_folder_name)
    if not os.path.exists(test_log_folder):
        os.mkdir(test_log_folder)
    return test_log_folder

if __name__ == '__main__':
    current_dir = os.getcwd()

    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)

    test_log_folder = get_save_log_path(grandparent_dir)

    monitor_log = os.path.join(grandparent_dir, "config", "resource_monitor.log")
    # monitor = ResourceMonitor(interval=2)  # 每2秒监控一次
    # monitor.start()
    config_file_path = os.path.join(grandparent_dir, "config", "test_config.ini")
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")
    lm = TinyLlamaLoader(config_file_path)
    # question = "简要解释量子力学"
    # lm.ask(question)

    grc = GradioClass(lm)
    grc.get_interface()
    grc.launch()


    # monitor.stop()
