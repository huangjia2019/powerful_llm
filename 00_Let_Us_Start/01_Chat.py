import openai

# openai.api_key = "your_api_key" 

completion = openai.chat.completions.create(
  model="gpt-4-turbo",
  messages=[
    {"role": "system", "content": "你是一位数据分析师,请根据以下步骤撰写一份数据分析报告。"},
    {"role": "user", "content": '''背景:电商平台要分析用户购物行为。请完成以下任务:
        1.提出2~3个切入点;
        2.列出每个切入点要分析的指标
        ;3.假设你发现了有价值的洞见,提出2~3条可行的建议
        ;4.综合成一份完整报告。'''
     }
  ],
  temperature=0.8
)

print(completion.choices[0].message.content)