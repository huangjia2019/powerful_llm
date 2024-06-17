import json
from collections import defaultdict
from openai import OpenAI

# 数据路径
data_path = "07_Finetune/Finetune_GPT3.5/finance_qa_data.jsonl"

# 加载数据集
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# 初始数据集统计
print("样本数量:", len(dataset))
print("第一个样本:")
for message in dataset[0]["messages"]:
    print(message)

# 格式错误检查
format_errors = defaultdict(int)

for ex in dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue
        
    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue
        
    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1
        
        if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
            format_errors["message_unrecognized_key"] += 1
        
        if message.get("role", None) not in ("system", "user", "assistant", "function"):
            format_errors["unrecognized_role"] += 1
            
        content = message.get("content", None)
        function_call = message.get("function_call", None)
        
        if (not content and not function_call) or not isinstance(content, str):
            format_errors["missing_content"] += 1
    
    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

if format_errors:
    print("发现错误:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("未发现错误")

# 如果没有错误,上传文件进行微调
if not format_errors:
    client = OpenAI()

    file = client.files.create(
        file=open(data_path, "rb"),
        purpose="fine-tune"
    )

    print(file)

'''
FileObject(id='file-ID8oHTZDz5jp4VdzOnJyvgfs', bytes=30946, created_at=1717608254, filename='finance_qa_data.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
'''