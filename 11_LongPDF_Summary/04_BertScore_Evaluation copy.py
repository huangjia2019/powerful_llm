# 读取摘要文件
def read_summary(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
t5_summary = read_summary("80_LLM实战课代码\\11_LongPDF_Summary\\t5_summary.txt")
claude_summary = read_summary("80_LLM实战课代码\\11_LongPDF_Summary\\claude_summary.txt")
ref_summary = read_summary("80_LLM实战课代码\\11_LongPDF_Summary\\ref_summary.txt")

# BERTScore评估
from bert_score import BERTScorer

scorer = BERTScorer(lang="en")
P1, R1, F1_t5 = scorer.score([t5_summary], [ref_summary])
P2, R2, F1_claude = scorer.score([claude_summary], [ref_summary])

# 打印BERTScore分数
print(f"\nT5 Summary BERTScore F1: {F1_t5.tolist()[0]}")
print(f"Claude Summary BERTScore F1: {F1_claude.tolist()[0]}")
