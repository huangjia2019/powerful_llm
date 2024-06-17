from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:personal::9Wou7j8z",
  messages=[
    {"role": "system", "content": "你是我的财务小助理"},
    {"role": "user", "content": "如何分析一支股票的基本面？"}
  ]
)
print(completion.choices[0].message)


