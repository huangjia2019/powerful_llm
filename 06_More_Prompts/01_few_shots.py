import os
from openai import OpenAI

client = OpenAI()

def get_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

few_shot_prompt = """
请对以下句子的情感进行分类,可以分为积极、中性或消极三类。

示例1:
句子: 这部电影真是太棒了,我非常喜欢!
情感: 积极

示例2:
句子: 天气还可以,不冷不热。
情感: 中性

示例3:
句子: 这家餐厅的服务太差了,再也不来了。
情感: 消极

句子: 这次旅行很不错,风景优美,吃的也不错。
情感:
"""

messages = [
    {"role": "system", "content": "你是一个情感分析模型,可以判断句子的情感是积极、中性还是消极。"},
    {"role": "user", "content": few_shot_prompt}
]

response = get_completion(messages)
print(response)