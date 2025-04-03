import gradio as gr

from src.model.load_model import TinyLlamaLoader
from src.utility.doc_reader import extract_pdf_text, extract_epub_text


class GradioClass():
    def __init__(self,lm):
        self.interface = None
        self.llm_loader = lm

    def launch(self):
        if not self.interface == None:
            self.interface.launch(server_port=7861)
            print("self.interface.launch() success")
        else:
            print("self.interface == None")
    def hello(self,message):
        return "hello, {}".format(message)
    def get_interface(self,function=None):
        # if function is None:
        #     function = self.hello  # 如果没有传入 function，默认使用 hello 函数
        # self.interface = gr.Interface(fn=function,
        #                          inputs=gr.File(file_types=[".pdf", ".epub"]),
        #                          outputs="text",
        #                          live=True)

        # self.interface = gr.Interface(fn=function,
        #     inputs=gr.Textbox(lines=3, placeholder="Name Here...",label="my input"),
        #     outputs="text",  # 输出文本框
        #     live=False,  # 关闭实时更新
        #     allow_flagging="never",  # 禁用标记按钮
        #     examples=[["这是一个示例文档内容，可以粘贴文本"]],  # 提供示例内容
        #     title="文档处理AI",  # 界面标题
        #     description="将文档内容粘贴到文本框，点击按钮进行处理。"  # 界面说明
        # )
        # 创建界面：上传文档 + 提问功能
        with gr.Blocks() as self.interface:
            with gr.Row():
                file_input = gr.File(label="上传文档")
                doc_output = gr.Textbox(label="文档加载反馈")
                file_input.upload(self.process_document, file_input, doc_output)

            with gr.Row():
                question_input = gr.Textbox(label="请输入问题", placeholder="你可以问：这批文档是关于什么的？", lines=2)
                submit_button = gr.Button("提交问题")
                answer_output = gr.Textbox(label="回答")
                # question_input.submit(self.ask_question, question_input, answer_output)
                submit_button.click(self.ask_question, inputs=question_input, outputs=answer_output)

    def process_document(self,file):
        return self.llm_loader.load_document(file)

    def ask_question(self,question):
        return self.llm_loader.answer_question(question)

