import pandas as pd
import numpy as np
import gensim
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from itertools import islice
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import pickle
import warnings
warnings.filterwarnings("ignore")

# 5. 构建数据加载器
class TextClassificationDataset(Dataset):
    def __init__(self, X, y, length):
        self.X = X
        self.y = y
        self.length = length
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.length[idx]

class GRUClassifierWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout, num_heads):
        super(GRUClassifierWithAttention, self).__init__()

        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2 if bidirectional else hidden_dim, num_heads=num_heads)
        self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.lstm(packed_sequence)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # 使用自注意力机制
        attention_output, attention_weights = self.attention(output.permute(1, 0, 2), output.permute(1, 0, 2), output.permute(1, 0, 2))
        attention_output = attention_output.permute(1, 0, 2)

        out = self.fc1(attention_output[:, -1, :])  # 使用最后一个时间步的输出
        out = self.tanh(out)
        out = self.fc2(out)
        
        return out, attention_weights

def train(model, train_loader, val_loader, num_epochs, learning_rate, save_path, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # 创建字典以存储训练和验证指标
    train_accuracies = []  # 存储训练准确率
    val_accuracies = []    # 存储验证准确率
    losses = []            # 存储损失
    
    best_val_accuracy = 0.0  # 用于跟踪最佳验证准确率
    best_model_state_dict = None  # 用于保存最佳模型参数
    consecutive_no_improvement = 0  # 用于跟踪连续没有提升的轮次
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 将模型移到GPU（如果可用）
    model.to(device)
    
    for epoch in range(num_epochs):
        if (epoch + 1) >= 10 and (epoch + 1) % 10 == 0:
            learning_rate = learning_rate / 10
            print('lr衰减:', learning_rate)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练循环
        for packed_inputs, labels, lengths in train_loader:
            packed_inputs, labels = torch.tensor(packed_inputs, dtype=torch.float32).to(device), (labels).to(device)
            lengths = lengths
            
            optimizer.zero_grad()
            
            outputs,weight= model(packed_inputs, lengths)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 计算训练集上的准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # 验证循环
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for packed_inputs, labels, lengths in val_loader:
                packed_inputs, labels = torch.tensor(packed_inputs.to(device), dtype=torch.float32), (labels).to(device)
                lengths = lengths.to('cpu')
                
                outputs,weight= model(packed_inputs, lengths)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 计算验证集上的准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        # 早停逻辑
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state_dict = model.state_dict()
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= patience:
            print(f"在{consecutive_no_improvement}轮次没有提升后进行早停。")
            break
    
    if best_model_state_dict is not None:
        # 保存最佳模型参数到文件
        torch.save(best_model_state_dict, save_path)
    
    print("训练结束!")
    
    metrics = {
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'losses': losses
    }

    return metrics

# 4. 读取预处理训练集、验证集和测试集
file_path = '/home/qiansongping/Experimental_Fish_3/6.train1/Vectors_21000'
train_X, train_y, train_length = torch.load(f'{file_path}/train_sentences_tensor.pth'),torch.load(f'{file_path}/train_labels_tensor.pth'), torch.load(f'{file_path}/train_lengths_tensor.pth')
val_X, val_y, val_length = torch.load(f'{file_path}/val_sentences_tensor.pth'),torch.load(f'{file_path}/val_labels_tensor.pth'), torch.load(f'{file_path}/val_lengths_tensor.pth')

batch_size = 4
train_dataset = TextClassificationDataset(train_X, train_y, train_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TextClassificationDataset(val_X, val_y, val_length)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


input_dim = 100  # 假设有200个词
hidden_dim = 8  # 隐藏层维度为8
output_dim = 2  # 假设二分类问题
num_layers = 1
bidirectional = True
dropout = 0.5
num_heads = 1

BiGRU_attention_model = GRUClassifierWithAttention(input_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout,num_heads)

# 调用训练函数并保存训练历史
num_epochs = 50
learning_rate = 0.01
save_path = '/home/qiansongping/Experimental_Fish_3/6.train1/Result21000/Best_GRU_attention_model05.pth'

metrics = train(BiGRU_attention_model, train_loader, val_loader, num_epochs, learning_rate, save_path)

with open('/home/qiansongping/Experimental_Fish_3/6.train1/Result21000/GRU_attention_training_metrics05.pkl', 'wb') as f:
    pickle.dump(metrics, f) 