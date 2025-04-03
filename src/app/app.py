import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_directml

# 初始化AMD DirectML设备
device = torch_directml.device() if torch_directml.is_available() else "cpu"

# 加载模型（4-bit量化节省显存）
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map=device,
    torch_dtype=torch.float16,
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def respond(message, history):
    # 构建对话历史
    prompt = "\n".join([f"You: {h[0]}\nAI: {h[1]}" for h in history] + [f"You: {message}\nAI: "])

    # 生成回复
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    return response


# 创建ChatUI
demo = gr.ChatInterface(
    respond,
    title="TinyLlama-1.1B 聊天助手 (AMD DirectML)",
    description="本地运行的TinyLlama模型，请用简短问题提问",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)  # 允许局域网访问