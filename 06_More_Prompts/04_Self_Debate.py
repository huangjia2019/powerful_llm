from openai import OpenAI

client = OpenAI()

def get_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message.content

def debate_and_reflect(question):
    initial_answer_prompt = f"""
    请解答以下问题:
    问题: {question}
    """
    
    messages = [
        {"role": "system", "content": "你是一个善于回答问题的助手。"},
        {"role": "user", "content": initial_answer_prompt}
    ]
    
    initial_answer = get_completion(messages)
    print("初始答案:", initial_answer)
    
    debate_prompt = f"""
    请针对以下问题和给出的答案,提出反对意见或质疑,并给出理由:
    问题: {question}
    答案: {initial_answer}
    """
    
    messages = [
        {"role": "system", "content": "你是一个善于从不同角度思考问题的助手。"},
        {"role": "user", "content": debate_prompt}
    ]
    
    debate = get_completion(messages)
    print("辩论意见:", debate)
    
    reflect_prompt = f"""
    请根据以下问题、最初给出的答案以及提出的质疑,进行反思并给出更全面的答案:
    问题: {question}
    最初答案: {initial_answer}
    质疑: {debate}
    """
    
    messages = [
        {"role": "system", "content": "你是一个善于反思和改进的助手。"},
        {"role": "user", "content": reflect_prompt}
    ]
    
    final_answer = get_completion(messages)
    print("反思后的答案:", final_answer)

question = "远程工作是否将成为大多数行业的常态?"
debate_and_reflect(question)
