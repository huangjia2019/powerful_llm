from openai import OpenAI
client = OpenAI()

# n_epochs = 2

# job = client.fine_tuning.jobs.create(
#   training_file="file-ID8oHTZDz5jp4VdzOnJyvgfs", 
#   model="gpt-3.5-turbo",
#   hyperparameters={"n_epochs": n_epochs}
# )

# print(job)

job = client.fine_tuning.jobs.retrieve('ftjob-b9S1AK4BHhBCYJv0I4LBZauM')

job_id = job.id
status = job.status

print(f"微调作业已创建,作业ID: {job_id}")

# 轮询作业状态,直至完成
import time
while status not in ["succeeded", "failed", "cancelled"]:
    print(f"作业状态: {status}, 等待 10 秒...")
    time.sleep(10)
    
    job = client.fine_tuning.jobs.retrieve(job_id)
    status = job.status

print(f"微调作业已完成,最终状态: {status}")

if status == "succeeded":
    print(f"微调后的模型名称: {job.fine_tuned_model}")

    # 输出微调信息
    response = client.fine_tuning.jobs.list_events(job_id)

    events = response.data
    events.reverse()

    for event in events:
        print(event.message)
else:
    print("微调作业未成功完成,请检查错误信息。")





'''
FineTuningJob(id='ftjob-b9S1AK4BHhBCYJv0I4LBZauM', created_at=1717608923, error=Error(code=None, message=None, param=None), fine_tuned_model=None, finished_at=None, hyperparameters=Hyperparameters(n_epochs=2, batch_size='auto', learning_rate_multiplier='auto'), model='gpt-3.5-turbo-0125', object='fine_tuning.job', organization_id='org-2MRF2ZhOMmubKIrnf84j3uGi', result_files=[], seed=1597839268, status='validating_files', trained_tokens=None, training_file='file-ID8oHTZDz5jp4VdzOnJyvgfs', validation_file=None, estimated_finish=None, integrations=[], user_provided_suffix=None)
微调作业已创建,作业ID: ftjob-b9S1AK4BHhBCYJv0I4LBZauM
'''