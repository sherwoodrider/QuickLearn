# 测试时添加资源监控
import psutil
def test_resource_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    print(cpu_percent)
    assert 40 < cpu_percent < 70  # 目标区间