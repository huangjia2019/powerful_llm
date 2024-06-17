from openai import OpenAI
client = OpenAI()

# 指定要查询的微调作业ID
job_id = 'ftjob-b9S1AK4BHhBCYJv0I4LBZauM'

# 查询微调作业
job = client.fine_tuning.jobs.retrieve(job_id)

# 输出微调信息
response = client.fine_tuning.jobs.list_events(job_id)

events = response.data
events.reverse()

for event in events:
    print(event.message)

# 获取结果文件ID
result_file = job.result_files[0]

# 获取结果文件信息
file = client.files.retrieve(result_file)
print(file)

# 获取结果文件内容
content = client.files.content(file.id)

content.stream_to_file('myfilename.txt')

# # 将二进制内容包装为类文件对象
# from io import BytesIO
# content_bytes = content.read()
# content_io = BytesIO(content_bytes)

# # 将CSV内容读取为数据框
# import pandas as pd
# df = pd.read_csv(content_io)

# # 打印数据框
# print(df)

