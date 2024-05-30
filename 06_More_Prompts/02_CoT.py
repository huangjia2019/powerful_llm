from openai import OpenAI

client = OpenAI()

def get_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

prompt = """
请解决以下应用题,并按照以下格式给出回答:

问题:
[问题描述]

解决:
1) [重述问题]
2) [解题步骤1]
3) [解题步骤2]
...

答案: [最终答案]

题目如下:
{question}
"""

def solve_problem(question):
    messages = [
        {"role": "system", "content": "你是一个擅长解决数学应用题的智能助手。"},
        {"role": "user", "content": prompt.format(question=question)}
    ]
    return get_completion(messages)

question = "一个水池有一进水管和一出水管。进水管每小时可注水180升,出水管每小时可排水120升。如果水池的容积是1200升,水池在开始注水时是空的,那么几小时后水池会注满水?"

result = solve_problem(question)
print(result)