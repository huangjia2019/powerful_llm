from openai import OpenAI

client = OpenAI()

def get_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.5,
        n=3,  # 生成3个候选答案
    )
    return [choice.message.content for choice in response.choices]

def self_consistency(question, solve_prompt):
    messages = [
        {"role": "system", "content": "你是一个善于解决问题的助手，解决问题方法不超过50字。"},
        {"role": "user", "content": solve_prompt.format(question=question)}
    ]
    
    candidate_answers = get_completion(messages)
    
    print(f"候选答案:\n{chr(10).join(candidate_answers)}\n")
    
    vote_prompt = f"""
    请对以下候选答案进行投票,选出最佳答案:
    
    候选答案:
    {chr(10).join(candidate_answers)}
    
    最佳答案是:"""
    
    messages = [
        {"role": "system", "content": "你是一个善于评判问题答案的助手,请返回最佳答案的阿拉伯数字编号。"},
        {"role": "user", "content": vote_prompt}
    ]

    best_answer_index = int(get_completion(messages)[0]) - 1
    best_answer = candidate_answers[best_answer_index]
    
    print(f"最佳答案的编号是: {best_answer_index + 1}")
    
    return best_answer

question = "马桶阻塞的最佳解决方案?"
solve_prompt = """
请解决以下应用题,并给出详细的解题步骤:
问题:
{question}
"""

result = self_consistency(question, solve_prompt)