import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig

MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, device):
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        formatted_example = self.format_example(example)
        inputs = self.tokenizer(
            formatted_example["context"],
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        labels = self.tokenizer(
            formatted_example["target"],
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        inputs['labels'] = labels['input_ids']
        result = {key: val.squeeze().to(self.device) for key, val in inputs.items()}
        
        # 打印输入数据和形状
        print(f"Sample {idx}:")
        print(f"Context: {formatted_example['context']}")
        print(f"Target: {formatted_example['target']}")
        print({key: val.shape for key, val in result.items()})
        
        return result

    def format_example(self, example: dict) -> dict:
        context = f"Instruction: {example['instruction']}\n"
        if example.get("input"):
            context += f"Input: {example['input']}\n"
        context += "Answer: "
        target = example["output"]
        return {"context": context, "target": target}

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 使用 BitsAndBytesConfig 进行量化配置
quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_threshold=200.0)
model = AutoModelForCausalLM.from_pretrained(MODEL, 
                                             trust_remote_code=True,
                                             device_map="auto",
                                             quantization_config=quantization_config,
                                             torch_dtype=torch.float16)

model.is_parallelizable = True
model.model_parallel = True
model.supports_gradient_checkpointing = True
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# 配置并应用 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # 确保这些模块名与 Llama3 兼容
)
model = get_peft_model(model, peft_config)

# 准备数据集
train_dataset = CustomDataset("data/chinese_med.json", tokenizer, device)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=5e-5,
    fp16=True,
    logging_dir='./logs',
    dataloader_pin_memory=False,
)

# 自定义 Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        print({key: val.shape for key, val in inputs.items()})  # 打印输入形状
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 定义 Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 开始训练
trainer.train()

# 创建保存模型的目录
import os
save_directory = 'Llama3-8B-Chat'
os.makedirs(save_directory, exist_ok=True)

# 保存训练后的模型和配置文件
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# 将配置文件下载到模型目录中
config = model.config.to_dict()
config_path = os.path.join(save_directory, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

print(f"Model and configuration saved to {save_directory}")
