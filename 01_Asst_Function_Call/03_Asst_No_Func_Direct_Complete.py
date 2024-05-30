# 导入OpenAI库,并创建OpenAI客户端
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()

# 检索您之前创建的Assistant
assistant_id = "asst_aT4hurwd35eSave7qrt2t6eJ"  # 你自己的助手ID
assistant = client.beta.assistants.retrieve(assistant_id)
print(assistant)

# 创建一个新的Thread
thread = client.beta.threads.create()
print(thread)

# 向Thread添加用户的消息
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="你好,请问你能做什么。"
)
print(message)

# 运行Assistant来处理Thread
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)
print("读取Run的状态", run)

import time
# 定义一个轮询的函数
def poll_run_status(client, thread_id, run_id, interval=10):
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Run的轮询信息:\n{run}\n")
        if run.status in ['requires_action', 'completed']:
            return run
        time.sleep(interval)  # 等待后再次检查

# 轮询以检查Run的状态
run = poll_run_status(client, thread.id, run.id)


# 获取Assistant在Thread中的回应
messages = client.beta.threads.messages.list(
    thread_id=thread.id
)
print("全部的message", messages)

# 输出Assistant的最终回应
print('下面打印最终的Assistant回应:')
for message in messages.data:
    if message.role == "assistant":
        print(f"{message.content}\n")