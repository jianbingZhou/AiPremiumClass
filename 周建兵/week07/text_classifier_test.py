import pickle
import torch
import torch.nn as nn
import jieba

class Comments_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, lbl_num):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, lbl_num)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out 
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import os
    os.chdir(os.path.dirname(__file__))

    embedding_dim = 100
    hidden_size = 128
    num_classes = 2

    # 加载词典
    vocab = torch.load('comments_vocab.pth')
    # 测试模型
    comment1 = '是一部难得的电影'
    comment2 = '演员演技在线'

    # 将评论转换为索引
    comment1_idx = torch.tensor([vocab.get(word, vocab['UNK']) for word in jieba.lcut(comment1)])
    comment2_idx = torch.tensor([vocab.get(word, vocab['UNK']) for word in jieba.lcut(comment2)])
    # 将评论转换为tensor
    comment1_idx = comment1_idx.unsqueeze(0).to(device)  # 添加batch维度    
    comment2_idx = comment2_idx.unsqueeze(0).to(device)  # 添加batch维度

    # 加载模型
    model = Comments_Classifier(len(vocab), embedding_dim, hidden_size, num_classes)
    model.load_state_dict(torch.load('comments_classifier.pth'))
    model.to(device)

    # 模型推理
    pred1 = model(comment1_idx)
    pred2 = model(comment2_idx)

    # 取最大值的索引作为预测结果
    pred1 = torch.argmax(pred1, dim=1).item()
    pred2 = torch.argmax(pred2, dim=1).item()
    pred1 =  '好评' if pred1 == 0 else '差评'
    pred2 =  '好评' if pred2 == 0 else '差评'
    print(f'评论1预测结果: {pred1}')
    print(f'评论2预测结果: {pred2}')  
           
