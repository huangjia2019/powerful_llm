# 读取摘要文件
def read_summary(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
t5_summary = read_summary("80_LLM实战课代码\\11_LongPDF_Summary\\t5_summary.txt")
claude_summary = read_summary("80_LLM实战课代码\\11_LongPDF_Summary\\claude_summary.txt")
ref_summary = read_summary("80_LLM实战课代码\\11_LongPDF_Summary\\ref_summary.txt")

from rouge import Rouge

# Rouge评估函数
def get_rouge_scores(summary, reference):
    rouge = Rouge()
    return rouge.get_scores(summary, reference)


# 打印ROUGE分数
rouge_scores_t5 = get_rouge_scores(t5_summary, ref_summary)
rouge_scores_claude = get_rouge_scores(claude_summary, ref_summary)

# 提取并显示关键的ROUGE指标
rouge_scores_out = []
for metric in ["rouge-1", "rouge-2", "rouge-l"]:
    for label in ["f", "p", "r"]:
        eval_1_score = rouge_scores_t5[0][metric][label]
        eval_2_score = rouge_scores_claude[0][metric][label]
        row = {
            "Metric": f"{metric} ({label})",
            "T5 Summary": eval_1_score,
            "Claude Summary": eval_2_score,
        }
        rouge_scores_out.append(row)

import pandas as pd
rouge_df = pd.DataFrame(rouge_scores_out).set_index("Metric")
print("ROUGE Scores:")
print(rouge_df)
