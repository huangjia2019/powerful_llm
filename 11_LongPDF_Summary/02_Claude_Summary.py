from dotenv import load_dotenv
load_dotenv()

import os 
import anthropic

# os.environ["ANTHROPIC_API_KEY"] = "<your_api_key>"

# 导入PyPDF2包
import PyPDF2

# 读取PDF文件内容的函数
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# 读取PDF内容
pdf_path = "99_data\\pdf\\2401.02385-TinyLlama.pdf"
pdf_text = read_pdf(pdf_path)

# 设计LLM总结论文提示词
def generate_summary(text):
    client = anthropic.Anthropic()
    
    system_prompt = "你是一个用于总结研究论文的AI助手。"
    
    messages = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": f"""
                    请总结以下研究论文,重点关注其关键发现、方法和结论:
                    
                    {text}
                    
                    请用大约150字提供一个简明扼要的英文摘要。
                    """
                }
            ]
        }
    ]
    
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=messages,
        system=system_prompt
    )
    
    return message.content[0].text.strip()

    
# 使用Claude生成总结
claude_summary = generate_summary(pdf_text)
print("Summary:", claude_summary)

# 保存生成的摘要到文件
with open("claude_summary.txt", "w", encoding="utf-8") as file:
    file.write(claude_summary)