# 导入PyPDF2包
import PyPDF2

# 读取PDF文件内容
pdf_path = "99_data\\pdf\\2401.02385-TinyLlama.pdf"
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# 读取PDF内容
pdf_text = read_pdf(pdf_path)

# 导入T5模型和分词器
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 定义函数来加载模型和分词器
def load_model(model_name='t5-small'):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# 定义函数进行文本总结
def summarize_text(text, tokenizer, model):
    # 使用T5的前缀来指定任务类型
    text = "summarize: " + text
    
    # 对文本进行编码
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    
    # 使用模型生成总结
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # 解码生成的总结
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# 加载预训练的T5模型
tokenizer, model = load_model('t5-small')

# 对PDF内容进行总结
t5_summary = summarize_text(pdf_text, tokenizer, model)
print("Generated Summary:", t5_summary)

# 保存生成的摘要到文件
with open("t5_summary.txt", "w", encoding="utf-8") as file:
    file.write(t5_summary)