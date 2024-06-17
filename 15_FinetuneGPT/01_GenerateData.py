from openai import OpenAI
from tqdm import tqdm
import json

client = OpenAI()

# 配置OpenAI API密钥
# openai.api_key = "your_api_key"  

# 定义问题生成函数
def generate_question(topic):
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": "你是一个善于提问的助手,会根据给定的话题生成问题。"},
            {"role": "user", "content": f"请问一个关于{topic}的问题,用于训练金融领域的聊天机器人。直接生成问题就好，不需要回复“好的”"}
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# 定义回答生成函数  
def generate_answer(question):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个金融领域的智能助理,能够专业地回答用户的金融问题。"},
            {"role": "user", "content": f"Q: {question}\nA: "}
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# 定义要生成问答数据的金融主题
topics = [
    "股票投资",
    "银行理财", 
    "基金定投",
    "资产配置",
    "风险管理"  
]

# 生成问答数据
qa_data = []
total_qa_pairs = len(topics) * 3
with tqdm(total=total_qa_pairs, desc="生成问答数据", unit="组") as pbar:
    for topic in topics:
        for i in range(3):  # 每个主题生成3组问答
            question = generate_question(topic)
            print(f"生成问题: {question}")
            
            answer = generate_answer(question)
            print(f"生成回答: {answer}\n")
            
            qa_data.append({
                "messages": [
                    {"role": "system", "content": "你是一个金融领域的智能助理,能够专业地回答用户的金融问题。"},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
            
            pbar.update(1)  # 更新进度条

# 将生成的数据保存到JSONL文件
with open("finance_qa_data.jsonl", "w", encoding="utf-8") as f:
    for data in qa_data:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"\n成功生成并保存{len(qa_data)}组金融问答数据到finance_qa_data.jsonl文件。")