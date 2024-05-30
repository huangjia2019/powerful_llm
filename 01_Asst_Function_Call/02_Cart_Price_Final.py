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
    content="你好,我购买了一本书和一个电子产品,请帮我计算一下订单总价。"
)
print(message)

# 运行Assistant来处理Thread
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)
print("读取function元数据信息之前Run的状态", run)

import time

# 定义一个轮询的函数
def poll_run_status(client, thread_id, run_id, interval=1):
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Run的轮询信息:\n{run}\n")
        if run.status in ['requires_action', 'completed']:
            return run
        time.sleep(interval)  # 等待后再次检查

# 轮询以检查Run的状态
run = poll_run_status(client, thread.id, run.id)

# 读取function元数据信息
def get_function_details(run):
    function_name = run.required_action.submit_tool_outputs.tool_calls[0].function.name
    arguments = run.required_action.submit_tool_outputs.tool_calls[0].function.arguments
    function_id = run.required_action.submit_tool_outputs.tool_calls[0].id
    return function_name, arguments, function_id

# 读取并打印元数据信息
function_name, arguments, function_id = get_function_details(run)
print("function_name:", function_name)
print("arguments:", arguments)
print("function_id:", function_id)

# 定义计算订单总价函数
def calculate_order_total(items):
    item_prices = {
        "书籍": 10,
        "文具": 5,
        "电子产品": 100
    }
    total_price = 0
    for item in items:
        price_per_item = item_prices.get(item['item_type'], 0)
        total_price += price_per_item * item['quantity']
    return total_price

# 根据Assistant返回的参数动态调用函数
import json

# 将 JSON 字符串转换为字典
arguments_dict = json.loads(arguments)

# 调用函数
order_total = globals()[function_name](**arguments_dict)

# 打印结果以进行验证
print(f"订单总价为: {order_total} 元")

# 提交结果
def submit_tool_outputs(run, thread, function_id, function_response):
    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=[
            {
                "tool_call_id": function_id,
                "output": str(function_response),
            }
        ]
    )
    return run

run = submit_tool_outputs(run, thread, function_id, order_total)
print("提交结果之后Run的状态", run)

# 再次轮询Run直至完成
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