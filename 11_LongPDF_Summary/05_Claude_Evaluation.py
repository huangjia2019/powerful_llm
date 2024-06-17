# 读取摘要文件
def read_summary(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
t5_summary = read_summary("80_LLM实战课代码\\11_LongPDF_Summary\\t5_summary.txt")
claude_summary = read_summary("80_LLM实战课代码\\11_LongPDF_Summary\\claude_summary.txt")
ref_summary = read_summary("80_LLM实战课代码\\11_LongPDF_Summary\\ref_summary.txt")

# Claude评估函数
import anthropic
import re
def get_claude_score(criteria, document, summary, metric_name):
    prompt = f"""
    你将会被提供一段文章和一段摘要。你的任务是根据以下标准对摘要进行评分（1到5分）：

    评价标准：
    {criteria}

    示例：
    原文：
    {document}

    摘要：
    {summary}

    请给出{metric_name}的评分（1到5分）："""

    client = anthropic.Anthropic()
    response = client.completions.create(
        prompt=anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=1000,
    )
    result = response.completion
    score = int(re.findall(r'\d+', result)[0])
    return score

evaluation_metrics = {
    "相关性": "相关性(1-5) - 摘要是否涵盖了原文中最重要和核心的信息。",
    "准确性": "准确性(1-5) - 摘要中的信息是否准确反映了原文的内容，是否存在误导或错误信息。",
    "简洁性": "简洁性(1-5) - 摘要是否在简明扼要的基础上传达了主要信息，而不是冗长或重复。",
    "流畅性": "流畅性(1-5) - 摘要的语言是否自然流畅，语法、拼写和标点是否正确。"
}

summaries = {"T5 Summary": t5_summary, "Claude Summary": claude_summary}

claude_scores = {"Metric": [], "T5 Summary": [], "Claude Summary": []}

for metric, criteria in evaluation_metrics.items():
    claude_scores["Metric"].append(metric)
    for summ_id, summ in summaries.items():
        score = get_claude_score(criteria, ref_summary, summ, metric)
        claude_scores[summ_id].append(score)

import pandas as pd
claude_df = pd.DataFrame(claude_scores).set_index("Metric")
print("\nClaude Evaluation Scores:")
print(claude_df)
