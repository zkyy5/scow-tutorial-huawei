from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
import os

# 模型和数据路径
model_path = "model"
data_path = "data"

# 加载模型
if os.path.exists(model_path) and os.path.isdir(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
else: 
    model = AutoModelForSequenceClassification.from_pretrained("./bert-base-uncased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")

# 加载数据
if os.path.exists(data_path) and os.path.isdir(data_path):
    raw_datasets = load_dataset(data_path)
else:
    raw_datasets = load_dataset("glue", "mrpc")

# 分词，使用 .map 方法为数据集添加 token 相关的 key
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 数据处理，去除不相关的 key，重命名 label key
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# 模型微调
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler, DataCollatorWithPadding
import torch
import torch_npu
from tqdm.auto import tqdm
import os
import evaluate
import numpy as np


# 指定 batch_size
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # 按照 batch 中最大的长度自动填充 padding
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# 硬件
device = torch.device("npu:0") if torch.npu.is_available() else torch.device("cpu")
model.to(device)

# 训练参数
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 评估函数
metric = evaluate.load("glue", "mrpc")
def compute_eval(model, eval_dataloader, device, metric=metric):
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
    
    eval_result = metric.compute()
    
    return eval_result

# 训练过程
progress_bar = tqdm(range(num_training_steps))
print('begin train', flush=True)
for epoch in range(num_epochs):
    model.train()
    print(f'begin epoch {epoch}', flush=True)
    for batch_idx, batch in enumerate(train_dataloader):
        print(f'begin batch_idx {batch_idx}', flush=True)
        # 前向传播
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        lr_scheduler.step()
        
        progress_bar.update(1)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}", flush=True)
    
    print(f"Epoch {epoch}: eval metric --> {compute_eval(model, eval_dataloader, device)}")

# 保存模型
model.save_pretrained('model_trained')
