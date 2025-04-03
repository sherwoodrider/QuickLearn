# 测试时添加资源监控
import psutil
# def test_resource_usage():
#     cpu_percent = psutil.cpu_percent(interval=1)
#     assert 40 < cpu_percent < 70  # 目标区间

import psutil
import time
import logging
from multiprocessing import Process, Event
from datetime import datetime


class ResourceMonitor:
    def __init__(self, log_file='resource_monitor.log', interval=1):
        """
        初始化资源监控器

        :param log_file: 日志文件路径
        :param interval: 监控间隔时间(秒)
        """
        self.log_file = log_file
        self.interval = interval
        self._stop_event = Event()
        self._process = None

        # 配置日志
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('ResourceMonitor')

    def _monitor(self):
        """监控循环"""
        self.logger.info("Resource monitoring started")

        while not self._stop_event.is_set():
            try:
                # 收集系统资源使用情况
                cpu_percent = psutil.cpu_percent(interval=self.interval)
                memory_info = psutil.virtual_memory()

                # import pyadl
                # 获取AMD显卡信息
                # adl = pyadl.ADLManager()
                # devices = adl.getDevices()
                # for i, device in enumerate(devices):
                #     print(f"GPU {i}: {device.name}")
                #     print(f"  Temperature: {device.getCurrentTemperature()}°C")
                #     print(f"  Usage: {device.getCurrentUsage()}%")
                #     print(f"  Fan Speed: {device.getCurrentFanSpeed()}%")
                #     print(f"  Core Clock: {device.getCurrentCoreClock()} MHz")
                #     print(f"  Memory Clock: {device.getCurrentMemoryClock()} MHz")

                # 记录到日志
                log_msg = (
                    f"CPU Usage: {cpu_percent:.1f}% | "
                    f"Memory: {memory_info.percent:.1f}% used "
                    f"({memory_info.used / (1024 ** 3):.2f} GB / "
                    f"{memory_info.total / (1024 ** 3):.2f} GB)"
                    # f"GPU: {devices[0].getCurrentUsage()}% used "
                )
                self.logger.info(log_msg)

            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                time.sleep(1)  # 出错时等待1秒再重试

    def start(self):
        """启动监控进程"""
        if self._process is None or not self._process.is_alive():
            self._stop_event.clear()
            self._process = Process(target=self._monitor)
            self._process.daemon = True  # 设置为守护进程
            self._process.start()

    def stop(self):
        """停止监控进程"""
        if self._process and self._process.is_alive():
            self._stop_event.set()
            self._process.join(timeout=5)
            self.logger.info("Resource monitoring stopped")

    def __del__(self):
        """析构函数确保监控停止"""
        self.stop()


# 使用示例
if __name__ == "__main__":
    # 创建并启动监控器
    monitor = ResourceMonitor(interval=2)  # 每2秒监控一次

    try:
        monitor.start()

        # 主程序工作...
        print("Main program is running... Press Ctrl+C to stop.")

        # 模拟主程序工作
        for i in range(10):
            # 这里可以放你的主程序代码
            print(f"Main program working... {i}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        monitor.stop()