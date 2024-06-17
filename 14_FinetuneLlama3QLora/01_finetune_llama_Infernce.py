# 导入所需的库
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 定义模型ID，这里使用我们微调过的Llama3-8B-Chat模型
model_id = "./Llama3-8B-Chat"

# 确保模型在单个设备上加载（如果有GPU，则使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=None,  # 不使用自动设备映射
    trust_remote_code=True  # 允许加载远程代码
)

# 将模型设置为评估模式
model.eval()

# 定义测试样本
test_examples = [
    {
        "instruction": "使用中医知识正确回答适合这个病例的中成药。",
        "input": "我前几天吃了很多食物，但肚子总是不舒服，咕咕响，还经常嗳气反酸，大便不成形，脸色也差极了。"
    },
    {
        "instruction": "使用中医知识正确回答适合这个病例的中成药。",
        "input": "肛门疼痛，痔疮，肛裂。"
    },
    {
        "instruction": "使用中医知识正确回答适合这个病例的中成药。",
        "input": "有没有能够滋养肝肾、清热明目的中药。"
    }
]

# 对每个测试样本生成回答
for example in test_examples:
    # 构建输入文本，包括指令和输入内容
    context = f"Instruction: {example['instruction']}\nInput: {example['input']}\nAnswer: "
    # 对输入文本进行分词
    inputs = tokenizer(context, return_tensors="pt")
    # 将输入数据移动到模型所在的设备上
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # 禁用FlashAttention以确保生成过程正常
    with torch.no_grad():  # 不计算梯度，节省内存
        outputs = model.generate(
            inputs['input_ids'],  # 输入的token ID
            max_length=512,  # 最大生成长度
            num_return_sequences=1,  # 只生成一个回答
            no_repeat_ngram_size=2,  # 避免重复生成n-gram
            use_cache=False  # 不使用缓存加速生成
        )

    # 解码生成的token ID，得到回答文本
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {example['input']}")
    print(f"Output: {answer}\n")