import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI()

# 加载图书数据集
dataset_path = "10_Book_Recommendation/图书.csv"
df = pd.read_csv(dataset_path)

# 选择需要的列
selected_columns = ['标题', '作者', '内容简介']
df = df[selected_columns].copy()

# 定义生成关键词的系统提示
keywords_system_prompt = '''
你是一个专门为图书标记相关关键词的代理。这些关键词可用于在图书馆或书店搜索这些书籍。

你将获得一本书的内容简介,你的目标是为这本书提取关键词。

关键词应简洁明了,全部小写。

关键词可以描述以下内容:  
- 书籍类型,如"科技"、"人文"、"心理学"等
- 书籍主题,如"机器学习"、"数据分析"、"自我提升"等  
- 目标读者,如"工程师"、"学生"、"普通人群"等
- 难度等级,如"初级"、"中级"、"高级"等

只有在书籍信息中明确提及时,才提取相关关键词。

以字符串数组的格式返回关键词,例如:
['心理学', '自我提升', '普通人群', '初级']
'''

# 定义生成关键词的函数
def generate_keywords(description):
    response = client.chat.completions.create(
    model="gpt-4",
    temperature=0.2,
    messages=[
        {
            "role": "system",
            "content": keywords_system_prompt
        },
        {
            "role": "user",
            "content": f"内容简介: {description}"
        }
    ],
    max_tokens=100,
    )
    return response.choices[0].message.content

# 为每本书生成关键词
df['关键词'] = df['内容简介'].apply(generate_keywords)

# 保存带有关键词的数据集
data_path = "10_Book_Recommendation/图书_带关键字.csv"
df.to_csv(data_path, index=False)

# 定义获取嵌入向量的函数
def get_embedding(value, model="text-embedding-ada-002"):
    embeddings = client.embeddings.create(
      model=model,
      input=value,
      encoding_format="float"
    )  
    return embeddings.data[0].embedding

# 嵌入标题、作者和关键词
df['embedding'] = df.apply(lambda x: get_embedding(f"{x['标题']} {x['作者']} {x['关键词']}"), axis=1)

# 将嵌入向量转换为字符串以便保存到CSV文件
df['embedding_str'] = df['embedding'].apply(lambda x: ','.join(map(str, x)))

# 保存带有关键词和嵌入向量的数据集
data_path = "10_Book_Recommendation/图书_词嵌入.csv"
df.to_csv(data_path, index=False)

# 从CSV文件加载带有关键词和嵌入向量的数据集
df_search = pd.read_csv(data_path)
df_search['embedding'] = df_search['embedding_str'].apply(lambda x: list(map(float, x.split(','))))

# 定义根据输入文本搜索的函数
def search_from_input_text(query, n=2):
    embedded_value = get_embedding(query)
    df_search['similarity'] = df_search['embedding'].apply(lambda x: cosine_similarity(np.array(x).reshape(1,-1), np.array(embedded_value).reshape(1, -1)))
    most_similar = df_search.sort_values('similarity', ascending=False).iloc[:n]
    return most_similar

# 测试输入文本搜索
user_input = "给我一本适合初学者入门的机器学习书"
res = search_from_input_text(user_input)
print(f"搜索词: {user_input}\n")
for index, row in res.iterrows():
    print(f"{row['标题']} ({row['作者']}) - 关键词: {row['关键词']}")