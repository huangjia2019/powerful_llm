# 导入所需的库
from io import BytesIO
import base64
from rich import print
import pandas as pd
import numpy as np

# 初始化OpenAI客户端
from openai import OpenAI
client = OpenAI()

# 指定要读取的单个PDF文件路径
file_path = "99_data\pdf\GPT图解.pdf"

# 提取文本内容
from pdfminer.high_level import extract_text

def extract_text_from_doc(path):
    """从PDF文件中提取文本内容"""
    text = extract_text(path)
    return text

text = extract_text_from_doc(file_path)

print(text)

# 将PDF转换为图像
from pdf2image import convert_from_path

def convert_doc_to_images(path):
    """将PDF文件转换为图像"""
    poppler_path = 'D:/venv/poppler/poppler-24.02.0/Library/bin'
    images = convert_from_path(path, poppler_path=poppler_path)
    return images

imgs = convert_doc_to_images(file_path)

# 将图像转换为base64编码的数据URI格式
def get_img_uri(img):
    """将图像转换为base64编码的数据URI格式"""
    buffer = BytesIO()
    img.save(buffer, format="jpeg")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_image}"
    return data_uri

# 定义系统提示,用于指导GPT-4V分析图像内容
system_prompt = '''
你将获得一张PDF页面或幻灯片的图片。你的目标是以技术术语描述你所看到的内容,就像你在进行演示一样。

如果有图表,请描述图表并解释其含义。
例如:如果有一个描述流程的图表,可以说"流程从X开始,然后有Y和Z..."

如果有表格,请从逻辑上描述表格中的内容。
例如:如果有一个列出项目和价格的表格,可以说"价格如下:A为X,B为Y..."  

不要包含涉及内容格式的术语。
不要提及内容类型,而是专注于内容本身。
例如:如果图片中有图表和文本,请同时描述两者,而不要提及一个是图表,另一个是文本。
只需描述你在图表中看到的内容以及从文本中理解到的内容。

你应该保持简洁,但请记住,你的听众看不到图片,所以要详尽地描述内容。

排除与内容无关的元素:
不要提及页码或图片上元素的位置。

------

如果有明确的标题,请按以下格式输出:

{标题}

{内容描述}  

如果没有明确的标题,只需返回内容描述即可。

'''

# 使用GPT-4V分析图像
def analyze_image(img_url):
    """使用GPT-4V分析图像"""
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        temperature=0,
        messages=[
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": img_url,
                    },  
                ],
            }
        ],
        max_tokens=300,
        top_p=0.1
    )

    return response.choices[0].message.content

# 分析PDF文档中的图像
def analyze_doc_image(img):
    """分析PDF文档中的图像"""
    img_uri = get_img_uri(img)
    data = analyze_image(img_uri)
    return data

pages_description = []

# 移除第一张幻灯片,通常只是简介
for img in imgs[1:]:
    res = analyze_doc_image(img)
    pages_description.append(res)

# 组合文本和图像分析结果
combined_content = []

# 去除第一张幻灯片
text_pages = text.split('\f')[1:]
description_indexes = []

for i in range(len(text_pages)): 
    slide_content = text_pages[i] + '\n'
    # 尝试找到匹配的幻灯片描述
    slide_title = text_pages[i].split('\n')[0]
    for j in range(len(pages_description)):
        description_title = pages_description[j].split('\n')[0]
        if slide_title.lower() == description_title.lower():
            slide_content += pages_description[j].replace(description_title, '')
            # 记录已添加的描述的索引
            description_indexes.append(j)
    # 将幻灯片内容和匹配的幻灯片描述添加到组合内容中 
    combined_content.append(slide_content)

# 添加未使用的幻灯片描述
for j in range(len(pages_description)):
    if j not in description_indexes:  
        combined_content.append(pages_description[j])

# 清理组合内容
import re

clean_content = []
for c in combined_content:
    text = c.replace(' \n', '').replace('\n\n', '\n').replace('\n\n\n', '\n').strip()
    text = re.sub(r"(?<=\n)\d{1,2}", "", text)
    text = re.sub(r"\b(?:the|this)\s*slide\s*\w+\b", "", text, flags=re.IGNORECASE)
    clean_content.append(text)

# 将清理后的内容转换为DataFrame  
df = pd.DataFrame(clean_content, columns=['content'])

# 获取嵌入向量
def get_embeddings(text):
    """获取给定文本的嵌入向量"""
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    )
    return embeddings.data[0].embedding

# 为每个内容片段生成嵌入向量
df['embeddings'] = df['content'].apply(lambda x: get_embeddings(x))

# 搜索相关内容
from sklearn.metrics.pairwise import cosine_similarity

def search_content(df, input_text, top_k):
    """搜索与输入文本最相关的内容"""
    embedded_value = get_embeddings(input_text)
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(np.array(x).reshape(1,-1), np.array(embedded_value).reshape(1, -1)))
    res = df.sort_values('similarity', ascending=False).head(top_k)
    return res

# 获取相似度得分
def get_similarity(row):
    """获取给定行的相似度得分"""
    similarity_score = row['similarity']
    if isinstance(similarity_score, np.ndarray):
        similarity_score = similarity_score[0][0]
    return similarity_score

# 定义系统提示,用于指导GPT-4回复输入查询
system_prompt = '''
    你将获得一个输入提示和一些作为上下文的内容,可以用来回复提示。
    
    你需要做两件事:
    
    1. 首先,你要在内部评估提供的内容是否与回答输入提示相关。
    
    2a. 如果内容相关,直接使用这些内容进行回答。如果内容相关,使用内容中的元素来回复输入提示。
    
    2b. 如果内容不相关,使用你自己的知识回答,如果你的知识不足以回答,就说你不知道如何回应。 
    
    保持回答简洁,具体回复输入提示,不要提及上下文内容中提供的额外信息。
'''

# 指定要使用的GPT-4模型
model="gpt-4-turbo-preview"  

# 生成输出
def generate_output(input_prompt, similar_content, threshold=0.5):
    """生成基于相似内容的输出"""
    content = similar_content.iloc[0]['content']
    
    # 如果相似度高于阈值,添加更多匹配的内容
    if len(similar_content) > 1:
        for i, row in similar_content.iterrows():
            similarity_score = get_similarity(row)
            if similarity_score > threshold:
                content += f"\n\n{row['content']}"
    
    prompt = f"输入提示:\n{input_prompt}\n-------\n内容:\n{content}"

    completion = client.chat.completions.create(
        model=model,
        temperature=0.5, 
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content

# 定义与内容相关的示例用户查询
example_inputs = [
    '语言模型的本质是什么?', 
    'NLP技术的目标是什么?',
    '语言模型经历了怎样的发展历程?',
    '作者解释语言时候，怎样用了古人进行对话?',
    '请介绍一下GPT图解一书作者?',
]

# 运行示例查询
for ex in example_inputs:
    print(f"[deep_pink4][bold]查询:[/bold] {ex}[/deep_pink4]\n\n")
    matching_content = search_content(df, ex, 3)
    print(f"[grey37][b]匹配内容:[/b][/grey37]\n")
    for i, match in matching_content.iterrows():
        print(f"[grey37][i]相似度: {get_similarity(match):.2f}[/i][/grey37]")
        content = str(match['content'])
        print(f"[grey37]{content[:100]}{'...' if len(content) > 100 else ''}[/[grey37]]\n\n")
    reply = generate_output(ex, matching_content)
    print(f"[turquoise4][b]回复:[/b][/turquoise4]\n\n[spring_green4]{reply}[/spring_green4]\n\n--------------\n\n")