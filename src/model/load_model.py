import configparser
import os
import torch_directml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from src.utility.doc_reader import extract_pdf_text, extract_epub_text


class TinyLlamaLoader():
    def __init__(self,config_path):
        self.config_path = config_path
        self.model_path = "D:/hugging_face/models/TinyLlama-1.1B"
        self.model_quant = "4bit" # 量化精度（4bit/8bit）
        self.model_max_memory = "6GB"  # 显存限制
        self.model_device = "auto"  # 自动选择GPU/CPU
        self.performance_batch_size = 1  # 单批次减少内存占用
        self.performance_torch_threads = 4  # 限制CPU线程数（i5-12600KF建议4-6）
        self.read_config()
        self.set_torch()
        self.tokenizer = None
        self.model = None
        self.gpu_device = torch_directml.device()
        self.text_data = ""  # 存储加载的文档文本
        # 加载模型
        self.get_model()
        self.get_tokenizer()

    def read_config(self):
        try:
            config = configparser.ConfigParser()
            # current_dir = os.getcwd()
            # config_file_path = os.path.join(current_dir, "src", "config", "test_config.ini")
            # if not os.path.exists(config_file_path):
            #     raise FileNotFoundError(f"Config file not found: {config_file_path}")
            print(self.config_path)
            config.read(self.config_path)
            self.model_path =  config['model']['path']
            self.model_quant =  config['model']['quant']
            self.model_max_memory =  config['model']['max_memory']
            self.model_device =  config['model']['device']
            self.performance_batch_size = int(config['performance']['batch_size'])
            self.performance_torch_threads =  int(config['performance']['torch_threads'])
        except Exception as e:
            print(e)

    def get_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def set_torch(self):
        # 性能优化设置
        torch.set_num_threads(self.performance_torch_threads)
        torch.backends.cudnn.benchmark = True  # 加速CUDA（如果可用）

    def get_model(self):
        try:
            # 加载模型（自动应用配置）
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cpu",
                # load_in_4bit=(self.model_quant == "4bit"),
                max_memory= {0: self.model_max_memory} if torch.cuda.is_available() else None
            )
            # self.model.eval()  # 切换到推理模式
            # self.model = torch.compile(self.model)  # PyTorch 2.0+ 编译优化（需CUDA）
        except Exception as e :
            print("get_model error : {}".format(str(e)))

    def ask(self,input_text = "解释量子力学："):
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt").to("cpu")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=200)
            print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        except Exception as e:
            print("ask error : {}".format(str(e)))

    def load_document(self,file):
        file_extension = file.name.split('.')[-1].lower()
        if file_extension == 'pdf':
            self.text_data = extract_pdf_text(file.name)
            return "文档加载完成！"
        elif file_extension == 'epub':
            self.text_data = extract_epub_text(file.name)
            return "文档加载完成！"
        else:
            return "仅支持PDF或EPUB格式的文件。"
    def answer_question(self, question):
        if not self.text_data:
            return "请先加载一个文档！"

        # 将文档内容与问题一起输入模型
        # input_text = f"文档内容：{self.text_data}\n问题：{question}"
        input_text = f"请阅读以下文档并回答问题：\n\n{self.text_data}\n\n问题：{question}\n回答："
        inputs = self.tokenizer(input_text, return_tensors="pt").to("cpu")

        # # 生成模型输出
        # with torch.no_grad():
        #     outputs = self.model.generate(**inputs, max_new_tokens=100)

        # 生成模型输出，调整生成参数
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,  # 可以适当增加生成长度
                temperature=0.7,  # 控制输出的多样性，较低值使模型更有确定性
                top_p=0.95,  # 限制生成的token范围
                do_sample=True  # 允许采样生成更具多样性的答案
            )
        print(outputs[0])
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


