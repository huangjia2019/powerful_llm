# 导入环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入OpenAI库,并创建OpenAI客户端
from openai import OpenAI

client = OpenAI()

assistant = client.beta.assistants.create(
    instructions="您是一个订单助手。请使用提供的函数来计算订单总价并回答问题。",
    model="gpt-4-1106-preview",
    tools=[{
        "type": "function",
        "function": {
            "name": "calculate_order_total",
            "description": "根据多个商品类型和数量计算订单总价",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_type": {
                                    "type": "string",
                                    "description": "商品类型,例如:书籍,文具,电子产品"
                                },
                                "quantity": {
                                    "type": "integer",
                                    "description": "商品数量"
                                }
                            },
                            "required": [
                                "item_type",
                                "quantity"
                            ]
                        }
                    }
                },
                "required": [
                    "items"
                ]
            }
        }
    }]
)

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