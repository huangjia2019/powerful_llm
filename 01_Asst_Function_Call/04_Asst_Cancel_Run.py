import time
from openai import OpenAI

# 创建OpenAI客户端
client = OpenAI()

# 创建一个新的Thread
thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": "你好,我购买了一本书、一支钢笔和一台笔记本电脑,请帮我计算一下订单总价。"
        }
    ]
)

# 在Thread上创建一个新的Run
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id="asst_aT4hurwd35eSave7qrt2t6eJ",  # 替换为你的Assistant ID
)

print(f"创建的Run: {run}")

# 轮询Run的状态,等待其进入in_progress状态
while run.status != "in_progress":
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    print(f"Run状态: {run.status}")
    time.sleep(1)

# 尝试取消正在进行中的Run
cancelled_run = client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
print(f"取消Run后的状态: {cancelled_run.status}")

# 继续轮询Run的状态,直到其变为最终状态(cancelled或failed)
while cancelled_run.status == "cancelling":
    cancelled_run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    print(f"取消Run后的状态: {cancelled_run.status}")
    time.sleep(1)

print(f"最终Run状态: {cancelled_run.status}")