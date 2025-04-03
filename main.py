import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pdfplumber import open as open_pdf
from sentence_transformers import SentenceTransformer


# 1. 文档加载与预处理（支持PDF/TXT）
file_paths = "./docs"
def load_docs(file_paths):
    texts = []
    for path in file_paths:
        if path.endswith(".pdf"):
            with open_pdf(path) as pdf:
                texts.append(" ".join(page.extract_text() for page in pdf.pages))
        else:  # TXT
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts


# 2. 语义搜索准备（轻量级模型）
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
documents = load_docs(["doc1.pdf", "doc2.txt"])
doc_embeddings = embedder.encode(documents)

# 3. 加载对话模型（4-bit量化）
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # 自动量化
    device_map="auto",  # 优先用GPU（AMD需安装ROCm）
    torch_dtype=torch.float16
)


# 4. 问答函数
def ask(question):
    # 先找到最相关文档段落
    question_embed = embedder.encode(question)
    similarity = torch.matmul(torch.tensor(question_embed), torch.tensor(doc_embeddings).T)
    most_relevant = documents[similarity.argmax().item()]

    # 生成回答
    prompt = f"根据以下内容回答问题：\n{most_relevant[:2000]}\n\n问题：{question}\n回答："
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe(prompt, max_length=512)[0]['generated_text']



# 5. 测试交互
print(ask("文档中提到的主要观点是什么？"))