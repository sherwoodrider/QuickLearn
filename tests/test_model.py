import pytest
from src.model.load_model import model, tokenizer
import time

@pytest.fixture
def test_cases():
    return [
        {"input": "1+1=", "min_len": 3},
        {"input": "Python的优点：", "keywords": ["简单", "易用"]},
    ]

def test_response_speed(test_cases):
    """测试响应时间"""
    for case in test_cases:
        start = time.time()
        inputs = tokenizer(case["input"], return_tensors="pt").to(model.device)
        _ = model.generate(**inputs, max_new_tokens=20)
        assert time.time() - start < 2.0  # 单次响应<2秒

def test_output_quality(test_cases):
    """测试输出质量"""
    for case in test_cases:
        inputs = tokenizer(case["input"], return_tensors="pt").to(model.device)
        output = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        if "keywords" in case:
            assert any(kw in output for kw in case["keywords"])
        if "min_len" in case:
            assert len(output) >= case["min_len"]

if __name__ == "__main__":
    pytest.main(["-v", "--durations=0"])