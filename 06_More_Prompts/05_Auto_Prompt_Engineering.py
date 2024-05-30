from openai import OpenAI

client = OpenAI()

def get_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def optimize_prompt(question, initial_prompt, iterations=3):
    current_prompt = initial_prompt
    
    for i in range(iterations):
        messages = [
            {"role": "system", "content": "你是一个擅长优化提示的助手。"},
            {"role": "user", "content": f"请优化以下提示,使其更适合解答特点领域的问题:\n\n当前提示:\n{current_prompt}\n\n问题:\n{question}"}
        ]
        
        current_prompt = get_completion(messages)
        
    return current_prompt

question = "我是一家餐馆的老板,最近有客人在就餐时不慎摔伤。我担心他会提出法律诉讼。在这种情况下,我有哪些法律责任?应该如何应对?"
initial_prompt = """
根据您提供的信息,我的建议如下:
首先,[...]
其次,[...]
再者,[...]
总之,[...]
如果您还有其他问题,欢迎随时向我咨询。
"""

optimized_prompt = optimize_prompt(question, initial_prompt)
print(f"优化后的提示:\n{optimized_prompt}")

# messages = [
#     {"role": "system", "content": "你是一个专业的法律顾问,善于就法律问题提供建议。"},
#     {"role": "user", "content": optimized_prompt.format(question=question)}
# ]

# result = get_completion(messages)
# print(f"\n使用优化后的提示得到的答案:\n{result}")