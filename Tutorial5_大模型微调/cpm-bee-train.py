from transformers import AutoTokenizer, AutoModelForCausalLM
from data_prepare import CPMDataset
from torch.utils.data import DataLoader
import torch
import time

torch.cuda.empty_cache()

# 数据准备
trainset = CPMDataset("bee_data/train.jsonl")  
train_loader = DataLoader(trainset, batch_size=1)
evalset = CPMDataset("bee_data/eval.jsonl")
eval_loader = DataLoader(evalset, batch_size=1)

# 模型
model_path = "cpm-bee-1b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters())

# 计时
total_time = 0

# 评估
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            input_encoded = tokenizer.prepare_for_finetune(data, max_length=512)
            input_encoded = {k: v.to(device) for k, v in input_encoded.items()}
            outputs = model(**input_encoded)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 训练
for iter, data in enumerate(train_loader):
    
    model.train()
    step_start = time.perf_counter()
    
    # 准备输入数据
    input_encoded = tokenizer.prepare_for_finetune(data, max_length=512)
    input_encoded = {k: v.to(device) for k, v in input_encoded.items()}
    
    # 前向传播
    outputs = model(**input_encoded)
    loss = outputs.loss

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    step_time = time.perf_counter() - step_start
    total_time += step_time
    
    # 100 个 batch 输出一次
    if iter % 100 == 0:
        print(f"Step {iter}, Loss: {loss.item():.4f}, Time per step: {step_time:.4f} s")
        eval_loss = evaluate(model, eval_loader)
        print(f"Validation Loss: {eval_loss:.4f}")

# 输出内容
print("Training done")
print(f"Total training time: {total_time:.2f} s")
print(f"Average time per step: {total_time / (iter + 1):.4f} s")
