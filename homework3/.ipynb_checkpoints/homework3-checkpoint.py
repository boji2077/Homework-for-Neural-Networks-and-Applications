# 导入依赖库
import json
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===================== 1. 超参数设置 =====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
EPOCHS = 2000
LR = 0.001
MAX_LEN = 28  # 七言绝句4句×7字=28字
START_WORD = "明月"  # 作业要求总起词

# ===================== 2. 数据预处理 =====================
def load_data(file_path):
    """加载古诗json数据，筛选七言绝句"""
    poems = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        paragraphs = item['paragraphs']
        # 筛选：2句、每句14字、无特殊符号
        if len(paragraphs) == 2:
            valid = True
            poem = ""
            for sent in paragraphs:
                sent = sent.strip()
                if len(sent) != 16:
                    valid = False
                    break
                sent = sent[:7] + sent[8:15]
                poem += sent
            poems.append(poem)
    return poems

def build_vocab(poems):
    """构建汉字-索引映射词典"""
    vocab = set()
    for poem in poems:
        vocab.update(poem)
    vocab = sorted(list(vocab))
    char2idx = {char: i+1 for i, char in enumerate(vocab)}  # 0留给填充
    char2idx['<PAD>'] = 0
    idx2char = {i: char for char, i in char2idx.items()}
    return char2idx, idx2char, len(char2idx)

def poem2tensor(poems, char2idx, max_len):
    """古诗转张量"""
    data = []
    for poem in poems:
        idx = [char2idx[c] for c in poem]
        # 填充/截断到固定长度
        if len(idx) < max_len:
            idx += [char2idx['<PAD>']] * (max_len - len(idx))
        else:
            idx = idx[:max_len]
        data.append(idx)
    data = np.array(data, dtype=np.int64)
    assert data.ndim == 2 and data.shape[1] == max_len, f"data shape error: {data.shape}"
    # 输入x=前n-1字，目标y=后n-1字
    x = torch.tensor(data[:, :-1], dtype=torch.long).to(DEVICE)
    y = torch.tensor(data[:, 1:], dtype=torch.long).to(DEVICE)
    return x, y

# 加载数据集（替换为你的poet.song路径）

poems = load_data("poet.song.40000.json")
print(f"加载到诗歌数量: {len(poems)}")
assert len(poems) > 0, "没有加载到任何诗歌，请检查数据文件或筛选条件！"
char2idx, idx2char, VOCAB_SIZE = build_vocab(poems)
x_data, y_data = poem2tensor(poems, char2idx, MAX_LEN)

# ===================== 3. LSTM模型定义 =====================
class PoemLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(PoemLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        out = self.fc(out)
        return out, hidden

# 初始化模型
model = PoemLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===================== 4. 模型训练 =====================
loss_history = []
print("========== 开始训练 ==========")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    # 分批训练
    for i in range(0, len(x_data), BATCH_SIZE):
        x_batch = x_data[i:i+BATCH_SIZE]
        y_batch = y_data[i:i+BATCH_SIZE]
        
        optimizer.zero_grad()
        output, _ = model(x_batch)
        # 调整维度计算损失
        loss = criterion(output.reshape(-1, VOCAB_SIZE), y_batch.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (len(x_data) // BATCH_SIZE + 1)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f}")

# ===================== 5. 绘制Loss曲线 =====================
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), loss_history, 'b-', linewidth=2)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.grid(True)
plt.savefig('loss_curve.png')  # 保存图片用于作业
plt.show()

# ===================== 6. 古诗生成（明月开头·七言绝句） =====================
def generate_poem(start_word, model, char2idx, idx2char, max_len=28):
    model.eval()
    poem = list(start_word)
    hidden = None
    with torch.no_grad():
        for _ in range(max_len - len(start_word)):
            x = torch.tensor([[char2idx[c] for c in poem]], dtype=torch.long).to(DEVICE)
            output, hidden = model(x, hidden)
            # 取最后一个字的预测概率
            prob = output.argmax(-1)[:, -1].item()
            poem.append(idx2char[prob])
    # 按7字一句拆分
    poem = ''.join(poem)
    return [poem[i*7:(i+1)*7] for i in range(4)]

# 生成古诗
print("\n========== 生成古诗 ==========")
result = generate_poem(START_WORD, model, char2idx, idx2char)
for idx, sent in enumerate(result, 1):
    print(f"{sent}")